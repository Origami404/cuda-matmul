#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <cublas_v2.h>

constexpr size_t N = 8192;
// how many threads in a block
auto constexpr BLK_N = 32;

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }
// use macro for both host and device
#define at(arr, i, j) ((arr)[(i)*N + (j)])

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *A, float *B) {
  auto const tx = blockIdx.x * BLK_N + threadIdx.x;
  auto const ty = blockIdx.y * BLK_N + threadIdx.y;

  __shared__ float sA[BLK_N][BLK_N + 1];
  __shared__ float sB[BLK_N][BLK_N + 1];
  float sum = 0.0f;

  for (auto b = 0; b < N; b += BLK_N) {
    sA[threadIdx.x][threadIdx.y] = at(A, tx, b + threadIdx.y);
    sB[threadIdx.x][threadIdx.y] = at(B, b + threadIdx.x, ty);

    __syncthreads();

    for (int k = 0; k < BLK_N; k++) {
      sum += sA[threadIdx.x][k] * sB[k][threadIdx.y];
    }
  }
  at(C, tx, ty) = sum;
}

// host copies of A, B, C
float hA[N * N], hB[N * N], hC[N * N];

auto test() {
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
  cudaMemset(C, 1, MAT_SIZE);

  //   dim3 constexpr blockDim{BLK_N, BLK_N, 1};
  //   dim3 constexpr gridDim{N / BLK_N, N / BLK_N, 1};

  cublasHandle_t handler;
  cublasCreate(&handler);

  auto const matmul_begin = std::chrono::steady_clock::now();

  //   matmul<<<gridDim, blockDim>>>(C, A, B);
  auto const alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handler, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N,
              &beta, C, N);

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
  auto constexpr TEST_N = 40;

  auto matmul_time = 0.0;
  for (auto i = 0; i < TEST_N; i++) {
    matmul_time += test();
  }
  matmul_time /= TEST_N;

  // each cell needs N fma, and there are N * N cells
  auto const matmul_tflops = 2.0 * N * N * N / matmul_time / 1e9;
  auto constexpr theory_max_tflops = 14.7456;

  std::cout << std::setprecision(3) << std::fixed;
  std::cout << "matmul time: " << matmul_time << "ms" << std::endl;
  std::cout << "Throughput: " << matmul_tflops << " TFLOPS "
            << "(" << matmul_tflops / theory_max_tflops * 100 << "%)"
            << std::endl;

  return 0;
}
