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
auto constexpr BK = 8;

auto constexpr TM_CNT = 16;
auto constexpr TN_CNT = 16;
// total 16*16 == 256 thread in a block

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
  static_assert(TM == TN && TN == BK && BK == 8, "use specified param");

  __shared__ float sA[BK * BM];
  __shared__ float sB[BK * BN];

  auto constexpr SMEM_USAGE = sizeof(sA) + sizeof(sB);
  static_assert(SMEM_USAGE <= 48 * 1024, "smem overflow");

  __align__(16) float pA[TM], pB[TN];
  float pC[TM][TN] = {0.0f};

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

  size_t Ay = tid / 2, Ax = (tid % 2) * 4;
  size_t By = tid / 32, Bx = (tid % 32) * 4;

  for (auto bk = 0; bk < K; bk += BK) {
    float *const gA = A + K * by + bk;
    float *const gB = B + N * bk + bx;

    float4 const rA = *as_f4p(gA + Ay * K + Ax);
    float4 const rB = *as_f4p(gB + By * N + Bx);
    // load A to smem, and transpose
    /*
     *   gA    8            sA    128
     *   +-----+-----+       +--+---------+---
     *   | t0  | t1  |       |t0|...      |padding
     *   +-----+-----+       |  |         |  |
     * 1 | t2  | t3  |     8 +--+---------+--+
     * 2 +-----+-----+       |t1|...      |padding
     * 8 | t4  | t5  |       |  |         |  |
     *   +-----+-----+       +--+---------+--+
     *   | ... | ... |
     *   |     |     |
     *   +-----+-----+
     *   |t254 |t255 |
     *   +-----+-----+
     */
    // 由于有 4 个 padding, t0 和 t1 的 bank conflict 并不会很严重

    sA[(Ax + 0) * BM + Ay] = rA.x;
    sA[(Ax + 1) * BM + Ay] = rA.y;
    sA[(Ax + 2) * BM + Ay] = rA.z;
    sA[(Ax + 3) * BM + Ay] = rA.w;

    // load B to smem directly
    /*
     *   gB       128
     *   +-----+-----+----------+-----+
     *   | t0  | t1  | ...      | t31 |
     * 8 +-----+-----+----------+-----+
     *   | ... |     |          |     |
     *   +-----+-----+----------+-----+
     *   | t224| t225| ...      | t255|
     *   +-----+-----+----------+-----+
     */
    *as_f4p(&sB[By * BN + Bx]) = rB;

    __syncthreads();

    for (size_t k = 0; k < BK; k++) {
      *as_f4p(pB + 0) = *as_f4p(&sB[k * BN + (sBx + 0)]);
      *as_f4p(pB + 4) = *as_f4p(&sB[k * BN + (sBx + 4)]);

      *as_f4p(pA + 0) = *as_f4p(&sA[k * BM + (sAy + 0)]);
      *as_f4p(pA + 4) = *as_f4p(&sA[k * BM + (sAy + 4)]);

#pragma unroll
      for (size_t y = 0; y < TM; y++) {
#pragma unroll
        for (size_t x = 0; x < TN; x++) {
          pC[y][x] += pA[y] * pB[x];
        }
      }
    }

    __syncthreads();
  }

  { // store pC back to gC
    for (size_t y = 0; y < TM; y++) {
      for (size_t x = 0; x < TN; x += 4) {
        *as_f4p(C + (Cy + y) * N + (Cx + x)) = *as_f4p(&pC[y][x]);
      }
    }
  }
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
