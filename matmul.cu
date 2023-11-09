#include <chrono>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <iomanip>
#include <iostream>
#include <random>

// just a constant
auto constexpr THREAD_PRE_WARP = 32;

// how many elements (float)
auto constexpr N = 8192;
auto constexpr M = 8192;
auto constexpr K = 8192;
// a N*K @ K*M matrix-matrix multiplication

// how many blocks in a grid

// how many elements in a thread-tile
auto constexpr TN = 2;
auto constexpr TM = 4;

// how many iters should a warp do for a warp-tile
auto constexpr TN_ITER = 2;
auto constexpr TM_ITER = 2;
// how many thread should a warp have
auto constexpr TN_CNT = 8;
auto constexpr TM_CNT = 4;
static_assert(TN_CNT * TM_CNT == THREAD_PRE_WARP, "");
// how many elements a warp-tile has
auto constexpr WN = TN_ITER * TN_CNT * TN;
auto constexpr WM = TM_ITER * TM_CNT * TM;

// how many elements should a block-tile have
auto constexpr BN = 64;
auto constexpr BM = 64;
auto constexpr BK = 32;

static_assert(BK >= WN && BK >= WM, "BK too small");
auto constexpr SMEM_SIZE = (BN * BK + BK * BM) * 4;
auto constexpr SMEM_MAX = 48 * 1024;
static_assert(SMEM_SIZE <= SMEM_MAX, "smem overflow");

// how many warp-tiles should a block have
auto constexpr WN_CNT = BN / WN;
auto constexpr WM_CNT = BM / WM;
// how many warps a block has,
// it should <= max active warps per SM, so SM can schedule them all.
auto constexpr WARP_PRE_BLK = WN_CNT * WM_CNT;
// it should <= max threads per block
auto constexpr THREAD_PRE_BLK = WARP_PRE_BLK * THREAD_PRE_WARP;

// how many block-tiles should a grid have
auto constexpr GN = N / BN;
auto constexpr GM = M / BM;

// theory metrics
auto constexpr BLK_PER_SM = SMEM_MAX / SMEM_SIZE;
auto constexpr BLK_CNT = GN * GM;

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B, int *debug) {
  *debug = -1; // if -1, the function return unexpectedly

  auto const thread_id = threadIdx.x;

  // each block has some smem cache
  __shared__ float sAr[BK][BN];
  __shared__ float sB[BK][BM];

  // each thread has some register cache
  float pA[TN][BK];
  float pB[BK][TM];
  float pC[TN][TM];

  for (auto bk_idx = 0; bk_idx < K / BK; bk_idx++) {
    // implicitly, we have two by/bx for loops here
    auto const by_idx = blockIdx.y;
    auto const bx_idx = blockIdx.x;

    // load A[by_idx * BN ...][bk_idx * BK ...] to sA[...][...]
    // load B[bk_idx * BK ...][bx_idx * BM ...] to sB[...][...]
    auto const by = by_idx * BN;
    auto const bx = bx_idx * BM;
    auto const bk = bk_idx * BK;

    // implicitly, we have two wy/wx for loops here
    auto const warp_id = thread_id / THREAD_PRE_WARP;
    auto const wy_idx = warp_id / WM_CNT;
    auto const wx_idx = warp_id % WM_CNT;
    // sA[wy_idx * WN ...][wx_idx * WM ...]
    // sB[wx_idx * WM ...][wy_idx * WN ...]

    auto const wy = wy_idx * WN;
    auto const wx = wx_idx * WM;

    // each thread should do TN_CNT * TM_CNT tiles
    auto const thread_wid = thread_id % THREAD_PRE_WARP;
    auto const thread_wid_y = thread_wid / TM_CNT;
    auto const thread_wid_x = thread_wid % TM_CNT;

    // load sA
    static_assert(BM % BK == 0 && TM_ITER % (BM / BK) == 0, "");
    auto constexpr TM_LOAD_ITER = TM_ITER / (BM / BK);
    auto const wxk = wx_idx * (TM_LOAD_ITER * TM_CNT * TM);

    for (auto ty_idx = 0; ty_idx < TN_ITER; ty_idx++) {
      for (auto tk_idx = 0; tk_idx < TM_LOAD_ITER; tk_idx++) {
        // tiles begins at (tk, tx)
        auto const ty = (ty_idx * TN_CNT + thread_wid_y) * TN;
        auto const tk = (tk_idx * TM_CNT + thread_wid_x) * TM;

        for (auto ey = 0; ey <= TN; ey++) {
          for (auto ex = 0; ex <= TM; ex++) {
            auto const sy = wy + ty + ey;
            auto const sk = wxk + tk + ex;
            sAr[sk][sy] = A[(by + sy) * K + (bk + sk)];
          }
        }
      }
    }

    // load sB
    static_assert(BN % BK == 0 && TN_ITER % (BN / BK) == 0, "");
    auto constexpr TN_LOAD_ITER = TN_ITER / (BN / BK);
    auto const wyk = wy_idx * (TN_LOAD_ITER * TN_CNT * TN);

    for (auto tk_idx = 0; tk_idx < TN_LOAD_ITER; tk_idx++) {
      for (auto tx_idx = 0; tx_idx < TM_ITER; tx_idx++) {
        // tiles begins at (ty, tx)
        auto const tk = (tk_idx * TN_CNT + thread_wid_y) * TN;
        auto const tx = (tx_idx * TM_CNT + thread_wid_x) * TM;

        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            auto const sk = wyk + tk + ey;
            auto const sx = wx + tx + ex;
            sB[sk][sx] = B[(bk + sk) * M + (bx + sx)];
          }
        }
      }
    }

    __syncthreads();

    if constexpr (true) {
      // check whether sA & sB is loaded correctly
      for (auto y = 0; y < BN; y++) {
        for (auto k = 0; k < BK; k++) {
          if (sAr[k][y] != A[(by + y) * K + k]) {
            *debug = 1;
            return;
          }
        }
      }

      for (auto k = 0; k < BK; k++) {
        for (auto x = 0; x < BM; x++) {
          if (sB[k][x] != B[(bk + k) * M + (bx + x)]) {
            *debug = 2;
            return;
          }
        }
      }
    }

    // for each warp iter, do k-times dot product
    for (auto ty_idx = 0; ty_idx < TN_ITER; ty_idx++) {
      for (auto tx_idx = 0; tx_idx < TM_ITER; tx_idx++) {
        // tiles begins at (ty, tx)
        auto const ty = (ty_idx * TN_ITER + thread_wid_y) * TN;
        auto const tx = (tx_idx * TM_ITER + thread_wid_x) * TM;

        // clear pC
        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            pC[ey][ex] = 0.0f;
          }
        }

        // load pA
        for (auto k = 0; k < BK; k++) {
          for (auto ey = 0; ey < TN; ey++) {
            auto const sy = wy + ty + ey;
            pA[ey][k] = sAr[k][sy];
          }
        }

        // load pB
        for (auto k = 0; k < BK; k++) {
          for (auto ex = 0; ex < TM; ex++) {
            auto const sx = wx + tx + ex;
            pB[k][ex] = sB[k][sx];
          }
        }

        // do dot product
        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            // k-times dot product
            for (auto k = 0; k < BK; k++) {
              pC[ey][ex] += pA[ey][k] * pB[k][ex];
            }
          }
        }

        // store pC
        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            auto const gy = by + wy + ty + ey;
            auto const gx = bx + wx + tx + ex;
            C[gy * M + gx] += pC[ey][ex];
          }
        }
      }
    }

    __syncthreads();
  }

  *debug = 0;
  return;
}

void run(float *dC, float *dA, float *dB) {
  int *deviceDebug;
  cudaMalloc(&deviceDebug, sizeof(int));
  cudaMemset(deviceDebug, 0, sizeof(int));

  dim3 constexpr blockDim{THREAD_PRE_BLK, 1, 1};
  dim3 constexpr gridDim{GN, GM, 1};
  matmul<<<gridDim, blockDim>>>(dC, dA, dB, deviceDebug);

  int hostDebug;
  cudaMemcpy(&hostDebug, deviceDebug, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(deviceDebug);

  if (hostDebug != 0) {
    std::cout << "Error debug: " << hostDebug << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }

// host copies of A, B, C
float hA[N * K], hB[K * M], hC[N * M], std_hC[N * M];
float *dA, *dB, *dC;

static inline float randf() {
  static std::random_device rd{};
  static std::mt19937 e{rd()};
  static std::uniform_real_distribution<float> d{0.0f, 1.0f};
  return d(e);
}

void mat_random(float *M, size_t n, size_t m) {
  for (auto y = 0; y < n; y++) {
    for (auto x = 0; x < m; x++) {
      M[y * m + x] = randf();
    }
  }
}

bool mat_eq(float *A, float *B, size_t n, size_t m) {
  for (auto y = 0; y < n; y++) {
    for (auto x = 0; x < m; x++) {
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
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dA, N, dB, K,
              &beta, dC, N);
  cudaMemcpy(std_hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

void init() {
  mat_random(hA, N, K);
  mat_random(hB, K, M);
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
  if (!mat_eq(hC, std_hC, N, M)) {
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
  auto constexpr WARMUP_N = 5;
  auto constexpr TEST_N = 30;

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
  auto const matmul_tflops = 2.0 * N * N * N / matmul_time / 1e9;
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
