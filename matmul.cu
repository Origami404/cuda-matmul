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

//=================================================//
// how many elements in a thread-tile
auto constexpr TN = 32;
auto constexpr TM = 1;

// how to place thread-tiles in a warp-tile
auto constexpr TN_CNT = 1;
auto constexpr TM_CNT = 32;

// how to place warp-tiles in a block-tile
auto constexpr WN_CNT = 2;
auto constexpr WM_CNT = 2;

// how many elements should a block-tile have
auto constexpr BN = 64;
auto constexpr BM = 64;
auto constexpr BK = 64;
//=================================================//

// warp-tile size
auto constexpr WN = TN * TN_CNT;
auto constexpr WM = TM * TM_CNT;

static_assert(BN % (WN * WN_CNT) == 0, "Unaligned BN");
static_assert(BM % (WM * WM_CNT) == 0, "Unaligned BM");
static_assert(BK % (WN * WN_CNT) == 0 && BK % (WM * WM_CNT) == 0,
              "Unaligned BK");
static_assert(WM % 4 == 0, "WM should be multiple of 4 to use vectorized load");

__global__ void matmul(float *C, float *A, float *B);
void run(float *dC, float *dA, float *dB) {
  auto constexpr THREAD_PRE_BLK = THREAD_PRE_WARP * WN_CNT * WM_CNT;
  auto constexpr GN = N / BN, GM = M / BM;

  dim3 constexpr blockDim{THREAD_PRE_BLK, 1, 1};
  dim3 constexpr gridDim{GM, GN, 1};
  matmul<<<gridDim, blockDim>>>(dC, dA, dB);
}

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B) {
  // each block has some smem cache
  __shared__ float sA[BN][BK];
  __shared__ float sB[BK][BM];
  __shared__ float sC[BN][BM];
  static_assert(sizeof(sA) + sizeof(sB) + sizeof(sC) <= 48 * 1024,
                "smem overflow");

  // each thread has some register cache
  float pC[TN][TM];

  // implicitly, we have two by/bx for loops here
  auto const by_idx = blockIdx.y;
  auto const bx_idx = blockIdx.x;

  auto const by = by_idx * BN;
  auto const bx = bx_idx * BM;

  // implicitly, we have two wy/wx for loops here
  auto const warp_id = threadIdx.x / THREAD_PRE_WARP;
  auto const wy_idx = warp_id / WM_CNT;
  auto const wx_idx = warp_id % WM_CNT;

  // implicitly, we have two ty/tx for loops here
  auto const thread_id = threadIdx.x % THREAD_PRE_WARP;
  auto const ty_idx = thread_id / TM_CNT;
  auto const tx_idx = thread_id % TM_CNT;

  auto const ty = ty_idx * TN;
  auto const tx = tx_idx * TM;

  { // load sC
    auto const ITER_Y = BN / (WN_CNT * WN), ITER_X = BM / (WM_CNT * WM);
    for (auto iter_y = 0; iter_y < ITER_Y; iter_y++) {
      for (auto iter_x = 0; iter_x < ITER_X; iter_x++) {
        auto const wy = (iter_y * WN_CNT + wy_idx) * WN;
        auto const wx = (iter_x * WM_CNT + wx_idx) * WM;

        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            auto const sy = wy + ty + ey;
            auto const sx = wx + tx + ex;
            sC[sy][sx] = C[(by + sy) * M + bx + sx];
          }
        }
      }
    }
  }
  __syncthreads();

  for (auto bk_idx = 0; bk_idx < K / BK; bk_idx++) {
    auto const bk = bk_idx * BK;

    { // load A to sA
      auto const ITER_Y = BN / (WN_CNT * WN), ITER_X = BK / (WM_CNT * WM);
      for (auto iter_y = 0; iter_y < ITER_Y; iter_y++) {
        for (auto iter_x = 0; iter_x < ITER_X; iter_x++) {
          auto const wy = (iter_y * WN_CNT + wy_idx) * WN;
          auto const wx = (iter_x * WM_CNT + wx_idx) * WM;

          for (auto ey = 0; ey < TN; ey++) {
            for (auto ex = 0; ex < TM; ex++) {
              auto const sy = wy + ty + ey;
              auto const sx = wx + tx + ex;

              sA[sy][sx] = A[(by + sy) * K + (bk + sx)];
            }
          }
        }
      }
    }

    { // load B to sB
      auto const ITER_Y = BK / (WN_CNT * WN), ITER_X = BM / (WM_CNT * WM);
      for (auto iter_y = 0; iter_y < ITER_Y; iter_y++) {
        for (auto iter_x = 0; iter_x < ITER_X; iter_x++) {
          auto const wy = (iter_y * WN_CNT + wy_idx) * WN;
          auto const wx = (iter_x * WM_CNT + wx_idx) * WM;

          for (auto ey = 0; ey < TN; ey++) {
            for (auto ex = 0; ex < TM; ex++) {
              auto const sy = wy + ty + ey;
              auto const sx = wx + tx + ex;

              sB[sy][sx] = B[(bk + sy) * M + (bx + sx)];
            }
          }
        }
      }
    }

    __syncthreads();

    { // compute C
      auto const ITER_Y = BN / (WN_CNT * WN), ITER_X = BM / (WM_CNT * WM);
      for (auto iter_y = 0; iter_y < ITER_Y; iter_y++) {
        for (auto iter_x = 0; iter_x < ITER_X; iter_x++) {
          auto const wy = (iter_y * WN_CNT + wy_idx) * WN;
          auto const wx = (iter_x * WM_CNT + wx_idx) * WM;

          // clear pC
          for (auto ey = 0; ey < TN; ey++) {
            for (auto ex = 0; ex < TM; ex++) {
              pC[ey][ex] = 0.0f;
            }
          }

          for (auto k = 0; k < BK; k++) {
            for (auto ey = 0; ey < TN; ey++) {
              for (auto ex = 0; ex < TM; ex++) {
                auto const sy = wy + ty + ey;
                auto const sx = wx + tx + ex;

                pC[ey][ex] += sA[sy][k] * sB[k][sx];
              }
            }
          }

          // store pC
          for (auto ey = 0; ey < TN; ey++) {
            for (auto ex = 0; ex < TM; ex++) {
              auto const sy = wy + ty + ey;
              auto const sx = wx + tx + ex;
              sC[sy][sx] += pC[ey][ex];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  { // store sC
    auto const ITER_Y = BN / (WN_CNT * WN), ITER_X = BM / (WM_CNT * WM);
    for (auto iter_y = 0; iter_y < ITER_Y; iter_y++) {
      for (auto iter_x = 0; iter_x < ITER_X; iter_x++) {
        auto const wy = (iter_y * WN_CNT + wy_idx) * WN;
        auto const wx = (iter_x * WM_CNT + wx_idx) * WM;

        for (auto ey = 0; ey < TN; ey++) {
          for (auto ex = 0; ex < TM; ex++) {
            auto const sy = wy + ty + ey;
            auto const sx = wx + tx + ex;
            C[(by + sy) * M + bx + sx] = sC[sy][sx];
          }
        }
      }
    }
  }
}

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-2f; }

// host copies of A, B, C
float hA[N * K], hB[K * M], hC[N * M], std_hC[N * M];
float *dA, *dB, *dC;

static inline float randf() {
  static std::random_device rd{};
  static std::mt19937 e{rd()};
  static std::uniform_real_distribution<float> d{-1.0f, 1.0f};
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
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K,
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
  auto const matmul_tflops = 2.0 * N * M * K / matmul_time / 1e9;
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
