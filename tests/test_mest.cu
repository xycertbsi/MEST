#include <mest/mest.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

static int passed = 0;
static int failed = 0;

#define TEST(name, expr) \
    do { \
        if (expr) { std::cout << "[PASS] " << name << "\n"; ++passed; } \
        else      { std::cout << "[FAIL] " << name << "\n"; ++failed; } \
    } while(0)

namespace kernels {
__global__ void add_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
}

void test_device() {
    int n = mest::Device::count();
    TEST("Device count >= 1", n >= 1);

    auto info = mest::Device::query(0);
    TEST("Device name non-empty", !info.name.empty());
    TEST("Device has memory", info.total_memory > 0);
}

void test_buffer() {
    std::vector<float> h = {1.f, 2.f, 3.f, 4.f};
    auto d = mest::upload(h);

    TEST("Buffer count", d.count() == 4);
    TEST("Buffer bytes", d.bytes() == 4 * sizeof(float));

    auto back = d.download();
    bool ok = true;
    for (size_t i = 0; i < h.size(); ++i) ok &= (back[i] == h[i]);
    TEST("Buffer round-trip", ok);
}

void test_kernel() {
    const int N = 512;
    std::vector<float> ha(N, 1.f), hb(N, 2.f);
    auto da = mest::upload(ha);
    auto db = mest::upload(hb);
    auto dc = mest::make_buffer<float>(N);

    auto cfg = mest::make_config(N);
    MEST_LAUNCH(kernels::add_f32, cfg, da.ptr(), db.ptr(), dc.ptr(), N);

    auto result = dc.download();
    bool ok = true;
    for (auto v : result) ok &= (std::fabs(v - 3.f) < 1e-5f);
    TEST("Kernel vector add", ok);
}

void test_stream() {
    mest::Stream s = mest::Stream::non_blocking();
    s.synchronize();
    TEST("Stream created and synced", true);
}

void test_context() {
    mest::Context ctx(0);
    ctx.synchronize();
    TEST("Context sync", true);
}

int main() {
    std::cout << "=== MEST Tests ===\n";
    test_device();
    test_buffer();
    test_kernel();
    test_stream();
    test_context();

    std::cout << "\n" << passed << " passed, " << failed << " failed.\n";
    return failed > 0 ? 1 : 0;
}
