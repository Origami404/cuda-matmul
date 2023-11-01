#include <cmath>
#include <cstddef>
#include <cstdio>

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

  matmul<<<gridDim, blockDim>>>(C, A, B);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaMemcpy(hC, C, MAT_SIZE, cudaMemcpyDeviceToHost);

  // Check for errors
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < N; j++) {
      auto const v = at(C, i, j);
      if ((i == j && !feq(v, 1.0f)) || (i != j && !feq(v, 0.0f))) {
        printf("Error: C[%d][%d] = %f\n", i, j, v);
      }
    }
  }

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
