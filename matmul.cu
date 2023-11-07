#include <chrono>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <iomanip>
#include <iostream>
#include <random>

constexpr size_t N = 8192;
auto constexpr BLK_N = 8;
auto constexpr THD_N = 8;
auto constexpr BLK_SIZE = BLK_N * THD_N;

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }
// use macro for both host and device
#define at(arr, i, j) ((arr)[(i)*N + (j)])
#define L(a, b, n) ((a) * (n) + (b))

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B) {
  __shared__ float sA[BLK_SIZE][BLK_SIZE];
  __shared__ float sB[BLK_SIZE][BLK_SIZE];
  float pA[THD_N], pB[THD_N];
  float pC[THD_N][THD_N] = {0.0f};

  auto const by = blockIdx.y;
  auto const bx = blockIdx.x;

  auto const ty = threadIdx.y;
  auto const tx = threadIdx.x;

  for (auto bk = 0; bk < N / BLK_SIZE; bk++) {
    // load sA/sB
    for (auto y = 0; y < THD_N; y++) {
      for (auto x = 0; x < THD_N; x += 4) {
        auto const gy = by * BLK_SIZE + ty * THD_N + y;
        auto const gk = bk * BLK_SIZE + tx * THD_N + x;
        float4 const t = *reinterpret_cast<float4 const *>(&at(A, gy, gk));

        auto const sy = ty * THD_N + y;
        auto const sx = tx * THD_N + x;
        sA[sy][sx + 0] = t.x;
        sA[sy][sx + 1] = t.y;
        sA[sy][sx + 2] = t.z;
        sA[sy][sx + 3] = t.w;
      }
    }

    for (auto y = 0; y < THD_N; y++) {
      for (auto x = 0; x < THD_N; x += 4) {
        auto const gk = bk * BLK_SIZE + ty * THD_N + y;
        auto const gx = bx * BLK_SIZE + tx * THD_N + x;
        float4 const t = *reinterpret_cast<float4 const *>(&at(B, gk, gx));

        auto const sy = ty * THD_N + y;
        auto const sx = tx * THD_N + x;
        sB[sy][sx + 0] = t.x;
        sB[sy][sx + 1] = t.y;
        sB[sy][sx + 2] = t.z;
        sB[sy][sx + 3] = t.w;
      }
    }

    // ensure sA/sB is loaded
    __syncthreads();

    for (auto tk = 0; tk < THD_N; tk++) {
      // load pA/pB
      for (auto y = 0; y < THD_N; y++) {
        pA[y] = sA[ty * THD_N + y][tx * THD_N + tk];
      }
      for (auto x = 0; x < THD_N; x++) {
        pB[x] = sB[ty * THD_N + tk][tx * THD_N + x];
      }

      // dot product
      for (auto y = 0; y < THD_N; y++) {
        for (auto x = 0; x < THD_N; x++) {
          pC[y][x] += pA[y] * pB[x];
        }
      }
    }

    // ensure sA/sB is no longger needed
    __syncthreads();
  }

  for (auto y = 0; y < THD_N; y++) {
    for (auto x = 0; x < THD_N; x += 4) {
      auto const gy = by * BLK_SIZE + ty * THD_N + y;
      auto const gx = bx * BLK_SIZE + tx * THD_N + x;
      float4 *gtp = reinterpret_cast<float4 *>(&at(C, gy, gx));
      float4 gt = *gtp;
      gt.x += pC[y][x + 0];
      gt.y += pC[y][x + 1];
      gt.z += pC[y][x + 2];
      gt.w += pC[y][x + 3];
      *gtp = gt;
    }
  }
}

// host copies of A, B, C
float hA[N * N], hB[N * N], hC[N * N], std_hC[N * N];

// host copies of A, B, C

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
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, dA, N, dB, N,
              &beta, dC, N);
  cudaMemcpy(std_hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

void init() {
  mat_random(hA, N, N);
  mat_random(hB, N, N);
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

  dim3 constexpr blockDim{BLK_N, BLK_N, 1};
  dim3 constexpr gridDim{N / BLK_SIZE, N / BLK_SIZE, 1};

  auto const matmul_begin = std::chrono::steady_clock::now();

  matmul<<<gridDim, blockDim>>>(C, A, B);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto const matmul_end = std::chrono::steady_clock::now();

  cudaMemcpy(hC, C, sizeof(hC), cudaMemcpyDeviceToHost);

  // Check for errors
  if (!mat_eq(hC, std_hC, N, N)) {
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
