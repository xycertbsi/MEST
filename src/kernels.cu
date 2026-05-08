#include "mest/kernel.h"
#include <cuda_runtime.h>

namespace mest {
namespace kernels {

__global__ void vector_add_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vector_scale_f32(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scalar;
}

__global__ void vector_add_f64(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matrix_transpose_f32(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        out[x * rows + y] = in[y * cols + x];
    }
}

__global__ void fill_f32(float* data, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

} // namespace kernels
} // namespace mest
