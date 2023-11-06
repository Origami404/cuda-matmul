#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

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
float hA[N * N], hB[N * N], hC[N * N];

auto test() {
  // init two matrixes
  for (auto y = 0; y < N; y++) {
    for (auto x = 0; x < N; x++) {
      at(hA, y, x) = y == x ? 1.0f : 0.0f;
    }
  }

  for (auto y = 0; y < N; y++) {
    for (auto x = 0; x < N; x++) {
      at(hB, y, x) = y == x ? 1.0f : 0.0f;
    }
  }

  // device copies of A, B, C
  float *A, *B, *C;

  // Allocate Unified Memory – accessible from CPU or GPU
  auto constexpr MAT_SIZE = N * N * sizeof(float);
  cudaMallocManaged(&A, MAT_SIZE);
  cudaMallocManaged(&B, MAT_SIZE);
  cudaMallocManaged(&C, MAT_SIZE);

  cudaMemcpy(A, hA, MAT_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(B, hB, MAT_SIZE, cudaMemcpyHostToDevice);
  cudaMemset(C, 0, MAT_SIZE);

  dim3 constexpr blockDim{BLK_N, BLK_N, 1};
  dim3 constexpr gridDim{N / BLK_SIZE, N / BLK_SIZE, 1};

  auto const matmul_begin = std::chrono::steady_clock::now();

  matmul<<<gridDim, blockDim>>>(C, A, B);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  auto const matmul_end = std::chrono::steady_clock::now();

  cudaMemcpy(hC, C, MAT_SIZE, cudaMemcpyDeviceToHost);

  // Check for errors
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      auto const v = at(C, i, j);
      if ((i == j && !feq(v, 1.0f)) || (i != j && !feq(v, 0.0f))) {
        std::cout << "Error at (" << i << ", " << j << "): " << v << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  auto const matmul_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(matmul_end -
                                                            matmul_begin)
          .count();

  return matmul_time;
}

int main(void) {
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
