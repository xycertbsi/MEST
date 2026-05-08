#include <mest/mest.h>
#include <iostream>
#include <vector>
#include <numeric>

namespace kernels {
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
}

int main() {
    const int N = 1024;

    std::vector<float> h_a(N), h_b(N);
    std::iota(h_a.begin(), h_a.end(), 0.0f);
    std::iota(h_b.begin(), h_b.end(), 1.0f);

    auto d_a = mest::upload(h_a);
    auto d_b = mest::upload(h_b);
    auto d_c = mest::make_buffer<float>(N);

    auto cfg = mest::make_config(N);
    MEST_LAUNCH(kernels::add, cfg, d_a.ptr(), d_b.ptr(), d_c.ptr(), N);

    auto result = d_c.download();

    std::cout << "Vector add — first 5 results:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << h_a[i] << " + " << h_b[i] << " = " << result[i] << "\n";
    }

    return 0;
}
