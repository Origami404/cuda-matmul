#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

constexpr size_t N = 8192;
// how many elements a thread handles (THD_N * THD_N)
auto constexpr THD_N = 8;
// how many threads in a block
auto constexpr BLK_N = 8;

static inline bool feq(float a, float b) { return abs(a - b) <= 1e-5f; }
// use macro for both host and device
#define at(arr, i, j) ((arr)[(i)*N + (j)])

__global__ void mat_transpose(float *A) {
  auto const tx = (blockIdx.x * BLK_N + threadIdx.x) * THD_N;
  auto const ty = (blockIdx.y * BLK_N + threadIdx.y) * THD_N;

  float pA[THD_N][THD_N];
  for (auto i = 0; i < THD_N; i++) {
    for (auto j = 0; j < THD_N; j++) {
      pA[i][j] = at(A, tx + i, ty + j);
    }
  }

  for (auto i = 0; i < THD_N; i++) {
    for (auto j = 0; j < i; j++) {
      pA[i][j] = pA[j][i];
    }
  }

  for (auto i = 0; i < THD_N; i++) {
    for (auto j = 0; j < THD_N; j++) {
      at(A, tx + i, ty + j) = pA[i][j];
    }
  }
}

// Kernel function to add the elements of two arrays
__global__ void matmul(float *C, float *Ar, float *B) {
  auto const tx_beg = (blockIdx.x * BLK_N + threadIdx.x) * THD_N;
  auto const ty_beg = (blockIdx.y * BLK_N + threadIdx.y) * THD_N;

  float pC[THD_N][THD_N], pA[THD_N], pB[THD_N];

  // set sC to 0
  for (auto i = 0; i < THD_N; i++) {
    for (auto j = 0; j < THD_N; j++) {
      pC[i][j] = 0.0f;
    }
  }

  for (auto k = 0; k < N; k++) {
    // load sA and sB
    for (auto y = 0; y < THD_N; y++) {
      pA[y] = at(Ar, k, ty_beg + y);
    }
    for (auto x = 0; x < THD_N; x++) {
      pB[x] = at(B, k, tx_beg + x);
    }

    // compute sC
    for (auto y = 0; y < THD_N; y++) {
      for (auto x = 0; x < THD_N; x++) {
        pC[y][x] += pA[y] * pB[x];
      }
    }
  }

  // write back to C
  for (auto y = 0; y < THD_N; y++) {
    for (auto x = 0; x < THD_N; x++) {
      at(C, ty_beg + y, tx_beg + x) = pC[y][x];
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
  cudaMemset(C, 1, MAT_SIZE);

  dim3 constexpr blockDim{BLK_N, BLK_N, 1};
  dim3 constexpr gridDim{N / BLK_N / THD_N, N / BLK_N / THD_N, 1};

  auto const matmul_begin = std::chrono::steady_clock::now();

  mat_transpose<<<gridDim, blockDim>>>(A);
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
  auto constexpr TEST_N = 15;

  auto matmul_time = 0.0;
  for (auto i = 0; i < TEST_N; i++) {
    matmul_time += test();
  }
  matmul_time /= TEST_N;

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
