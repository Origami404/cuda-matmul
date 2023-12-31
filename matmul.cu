#include <chrono>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <iomanip>
#include <iostream>
#include <random>

// how many elements (float)
auto constexpr M = 8192;
auto constexpr N = 8192;
auto constexpr K = 8192;
// a M*K @ K*N matrix-matrix multiplication

auto constexpr TM = 8;
auto constexpr TN = 8;

auto constexpr BM = 128;
auto constexpr BN = 128;
auto constexpr BK = 16;

auto constexpr TM_CNT = 16;
auto constexpr TN_CNT = 16;
// total 16*16 == 256 thread in a block
auto constexpr BLK_THREAD_NUM = TM_CNT * TN_CNT;

__global__ void matmul(float *C, float *A, float *B);
void run(float *C, float *A, float *B) {
  dim3 constexpr blockDim{TM_CNT * TN_CNT, 1, 1};
  dim3 constexpr gridDim{M / BM, N / BN, 1};
  matmul<<<gridDim, blockDim>>>(C, A, B);
}

__device__ __forceinline__ float4 *as_f4p(float *p) {
  return reinterpret_cast<float4 *>(p);
}

__global__ void matmul(float *C, float *A, float *B) {
  auto const tid = threadIdx.x;
  auto const by = blockIdx.y * BM;
  auto const bx = blockIdx.x * BN;

  { // How to tile threads when computing gC to avoid bank conflict
    /*
     * gC                       2 warps
     *   +--+--+--+--+--+--+--+--+  +-----------------------+
     *   | 0| 4| 8|12|16|20|24|28|  |32 ...                 |
     *   +--+--+--+--+--+--+--+--+  |                       |
     *   | 1| 5|  |  |  |  |  |  |  |                       |
     *   +--+--+--+--+--+--+--+--+  |                       |
     *   | 2| 6|  |  |  |  |  |  |  |                       |
     *   +--+--+--+--+--+--+--+--+  |                       |
     *   | 3| 7|  |  |  |  |  |31|  |                     63|
     * 4 +--+--+--+--+--+--+--+--+  +-----------------------+
     *             .                          .
     * w           .                          .
     * a           .                          .
     * r           .                          .
     * p +-----------------------+  +-----------------------+
     * s |                       |  |224 ...                |
     *   |                       |  |                       |
     *   |                       |  |                       |
     *   |                       |  |                       |
     *   |                       |  |                       |
     *   |                       |  |                       |
     *   |                       |  |                    255|
     *   +-----------------------+  +-----------------------+
     */
  }
  size_t const wid = tid / 32;
  size_t const wy = (wid / 2) * (4 * TM);
  size_t const wx = (wid % 2) * (8 * TN);

  size_t const ttid = tid % 32;
  size_t const ty = (ttid % 4) * TM;
  size_t const tx = (ttid / 4) * TN;

  size_t const sAy = wy + ty;
  size_t const sBx = wx + tx;
  size_t const Cy = by + wy + ty;
  size_t const Cx = bx + wx + tx;

  { // load A to smem, and transpose
    /*
     *   gA    BK           sA    128
     *   +-----+-----+       +--+---------+
     *   | t0  |t1   |       |t0|...      |
     *   +-----+-----+       |  |         |
     * 1 | t2  |t3   |     B +--+---------+
     * 2 +-----+-----+     K |t1|...      |
     * 8 | ... | ... |       |  |         |
     *   |     |     |       +--+---------+
     *   +-----+-----+
     *   |t254 |t255 |
     *   +-----+-----+
     */
  }
  size_t constexpr TX_GA = (BK / 2);
  size_t Ay = tid / 2, Ax = (tid % 2) * TX_GA;

  { // load B to smem directly
    /*
     *   gB       128
     *   +-----+-----+----------+-----+
     *   | t0  | t1  | ...      | t31 | TY_GB
     * B +-----+-----+----------+-----+
     * K | ... |     |          |     |
     *   +-----+-----+----------+-----+
     *   | t224| t225| ...      | t255| TY_GB
     *   +-----+-----+----------+-----+
     */
  }
  size_t constexpr TY_GB = (BK / (BLK_THREAD_NUM / 32));
  size_t By = (tid / 32) * TY_GB, Bx = (tid % 32) * 4;

  __shared__ float sA[BK * BM];
  __shared__ float sB[BK * BN];

  auto constexpr SMEM_USAGE = sizeof(sA) + sizeof(sB);
  static_assert(SMEM_USAGE <= 48 * 1024, "smem overflow");

  // registers used by compution
  __align__(16) float pA[2][TM], pB[2][TN];
  float pC[TM][TN] = {0.0f};

  // registers used by gmem -> smem
  float4 rA[TX_GA / 4];
  float4 rB[TY_GB];

  auto const gmem_load = [&](size_t const &bk) {
    float *const gA = A + K * by + bk;
    for (size_t x = 0; x < TX_GA; x += 4) {
      rA[x / 4] = *as_f4p(gA + Ay * K + (Ax + x));
    }
    float *const gB = B + N * bk + bx;
    for (size_t y = 0; y < TY_GB; y++) {
      rB[y] = *as_f4p(gB + (By + y) * N + Bx);
    }
  };

  auto const smem_store = [&]() {
    for (size_t x = 0; x < TX_GA; x += 4) {
      sA[(Ax + x + 0) * BM + Ay] = rA[x / 4].x;
      sA[(Ax + x + 1) * BM + Ay] = rA[x / 4].y;
      sA[(Ax + x + 2) * BM + Ay] = rA[x / 4].z;
      sA[(Ax + x + 3) * BM + Ay] = rA[x / 4].w;
    }
    for (size_t y = 0; y < TY_GB; y++) {
      *as_f4p(&sB[(By + y) * BN + Bx]) = rB[y];
    }
  };

  auto const smem_load = [&](size_t const &k) {
    size_t const iter = k % 2;

    *as_f4p(pB[iter] + 0) = *as_f4p(&sB[k * BN + (sBx + 0)]);
    *as_f4p(pB[iter] + 4) = *as_f4p(&sB[k * BN + (sBx + 4)]);

    *as_f4p(pA[iter] + 0) = *as_f4p(&sA[k * BM + (sAy + 0)]);
    *as_f4p(pA[iter] + 4) = *as_f4p(&sA[k * BM + (sAy + 4)]);
  };

  auto const compute = [&](size_t const &k) {
    size_t const iter = k % 2;

#pragma unroll
    for (size_t y = 0; y < TM; y++) {
#pragma unroll
      for (size_t x = 0; x < TN; x++) {
        pC[y][x] += pA[iter][y] * pB[iter][x];
      }
    }
  };

  auto const gmem_store = [&]() {
    for (size_t y = 0; y < TM; y++) {
      for (size_t x = 0; x < TN; x += 4) {
        *as_f4p(C + (Cy + y) * N + (Cx + x)) = *as_f4p(&pC[y][x]);
      }
    }
  };

  { // how double buffering works
    // clang-format off
    // g>i : gmem_load(i * BK), s<i : smem_store(i * BK)
    // s>ij : smem_load(j), cij : compute(j)
    // Basicly, `>` for load, `<` for store. The | line for __syncthreads().
/*
*                         outer-loop k=BK                                                                            outer-loop final iter
*          <---------------------------------------------->                                                        <----------------------->
* 
*         |                                            |   |                                            |   |     |                 |       |
* g>0 s<0 | g>1                                        |s<1| g>2                                        |s<2|     | g>(K-1)         |s<(K-1)|
*         |                                            |   |                                            |   |     |                 |       |
*         | +----------------------------------------+ |   | +----------------------------------------+ |   | ... | +-------------+ |       |  +-------------+
*         | |s>00  s>01  s>02 ...  s>0(k-1)          | |   | |s>10  s>11  s>12 ...  s>1(k-1)          | |   |     | |...          | |       |  |...          |
*         | |      c00   c01  ...  c0(k-2)   c0(k-1) | |   | |      c10   c11  ...  c1(k-2)   c1(k-1) | |   |     | |             | |       |  |             |
*         | +----------------------------------------+ |   | +----------------------------------------+ |   |     | +-------------+ |       |  +-------------+
*         |  inner-loop 0                    BK times  |   |  inner-loop 1                    BK times  |   |     | inner-loop (K-2)|       |  inner-loop (K-1)
*/
    // clang-format on
  }

  const auto inner_loop = [&]() {
    smem_load(0);
#pragma unroll
    for (size_t k = 1; k < BK; k++) {
      // this two stage can be parallel in hardware
      smem_load(k);
      compute(k - 1);
    }
    compute(BK - 1);
  };

  // outer-compute-loop
  gmem_load(0);
  smem_store();
  __syncthreads();

  for (auto bk = BK; bk < K; bk += BK) {
    // this two stage can be parallel in hardware
    gmem_load(bk);
    inner_loop();
    __syncthreads();
    smem_store();
    __syncthreads();
  }

  inner_loop();

  // finally, store back to C
  gmem_store();
}

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-2f; }

// host copies of A, B, C
float hA[M * K], hB[K * N], hC[M * N], std_hC[M * N];
float *dA, *dB, *dC;

static inline float randf() {
  static std::random_device rd{};
  static std::mt19937 e{rd()};
  static std::uniform_real_distribution<float> d{-1.0f, 1.0f};
  return d(e);
}

void mat_random(float *M, size_t n, size_t m) {
  for (size_t y = 0; y < n; y++) {
    for (size_t x = 0; x < m; x++) {
      M[y * m + x] = randf();
    }
  }
}

bool mat_eq(float *A, float *B, size_t n, size_t m) {
  for (size_t y = 0; y < n; y++) {
    for (size_t x = 0; x < m; x++) {
      if (!feq(A[y * m + x], B[y * m + x])) {
        std::cout << "A[" << y << "][" << x << "] = " << A[y * m + x]
                  << " != B[" << y << "][" << x << "] = " << B[y * m + x]
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

void calc_std() {
  float *dA, *dB, *dC;

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaMalloc(&dA, sizeof(hA));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);

  cudaMalloc(&dB, sizeof(hB));
  cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

  cudaMalloc(&dC, sizeof(hC));
  cudaMemset(dC, 0, sizeof(hC));
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dB, M, dA, K,
              &beta, dC, M);
  cudaMemcpy(std_hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

void init() {
  mat_random(hA, M, K);
  mat_random(hB, K, N);
  calc_std();
}

auto test() {
  // device copies of A, B, C
  float *A, *B, *C;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, sizeof(hA));
  cudaMalloc(&B, sizeof(hB));
  cudaMalloc(&C, sizeof(hC));

  cudaMemcpy(A, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB, sizeof(hB), cudaMemcpyHostToDevice);
  cudaMemset(C, 0, sizeof(hC));

  auto const matmul_begin = std::chrono::steady_clock::now();
  run(C, A, B);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto const matmul_end = std::chrono::steady_clock::now();

  cudaMemcpy(hC, C, sizeof(hC), cudaMemcpyDeviceToHost);

  // Check for errors
  if (!mat_eq(hC, std_hC, M, N)) {
    std::cout << "Error!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  using std::chrono::duration_cast;
  using std::chrono::milliseconds;
  return duration_cast<milliseconds>(matmul_end - matmul_begin).count();
}

int main(void) {
  init();
  std::cout << "Finish init" << std::endl;

#ifndef PROFILE
  auto constexpr WARMUP_N = 10;
  auto constexpr TEST_N = 40;

  for (auto i = 0; i < WARMUP_N; i++) {
    test();
  }

  auto matmul_time = 0.0;
  for (auto i = 0; i < TEST_N; i++) {
    matmul_time += test();
  }
  matmul_time /= TEST_N;
#else
  auto const matmul_time = test();
#endif

  // each cell needs N fma, and there are N * N cells
  auto const matmul_tflops = 2.0 * M * N * K / matmul_time / 1e9;
  auto constexpr theory_max_tflops = 14.7456;
  auto constexpr cublas_tflops = 8.228;

  std::cout << std::setprecision(3) << std::fixed;
  std::cout << "matmul time: " << matmul_time << " ms" << std::endl;
  std::cout << "Throughput: " << matmul_tflops << " TFLOPS \n"
            << "    (" << matmul_tflops / theory_max_tflops * 100 << "% Max)\n"
            << "    (" << matmul_tflops / cublas_tflops * 100 << "% cuBLAS)\n"
            << std::endl;

  return 0;
}
