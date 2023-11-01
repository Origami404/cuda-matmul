#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

constexpr size_t N = 8192;

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }
// use macro for both host and device
#define at(arr, i, j) ((arr)[(i)*N + (j)])

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B) {
  auto const tx = blockIdx.x * blockDim.x + threadIdx.x;
  auto const ty = blockIdx.y * blockDim.y + threadIdx.y;

  at(C, tx, ty) = 0.0f;
  for (auto i = 0; i < N; i++) {
    at(C, tx, ty) += at(A, tx, i) * at(B, i, ty);
  }
}

// host copies of A, B, C
float hA[N * N], hB[N * N], hC[N * N];

int main(void) {
  // init two matrixes
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      at(hA, i, j) = i == j ? 1.0f : 0.0f;
    }
  }

  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      at(hB, i, j) = i == j ? 1.0f : 0.0f;
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

  // Run kernel on 1M elements on the GPU
  auto constexpr BLK_N = 16;
  dim3 const blockDim{BLK_N, BLK_N, 1};
  dim3 const gridDim{N / BLK_N, N / BLK_N, 1};

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
      }
    }
  }

  auto const matmul_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(matmul_end -
                                                            matmul_begin)
          .count();

  // each cell needs N fma, and there are N * N cells
  auto const matmul_tflops = 2.0 * N * N * N / matmul_time / 1e9;
  auto constexpr theory_max_tflops = 14.7456;

  std::cout << std::setprecision(3) << std::fixed;
  std::cout << "matmul time: " << matmul_time << "ms" << std::endl;
  std::cout << "Throughput: " << matmul_tflops << " TFLOPS "
            << "(" << matmul_tflops / theory_max_tflops * 100 << "%)"
            << std::endl;

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
