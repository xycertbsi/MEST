#include "mest/capi.h"
#include "mest/context.h"
#include "mest/memory.h"
#include "mest/kernel.h"
#include "mest/error.h"

#include <string>
#include <cstring>

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

struct F32Buffer { mest::Buffer<float> buf; size_t count; };
struct F64Buffer { mest::Buffer<double> buf; size_t count; };

} // namespace

extern "C" {

mest_context_t mest_context_create(int device_id) {
    try {
        return new mest::Context(device_id);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void mest_context_destroy(mest_context_t ctx) {
    delete static_cast<mest::Context*>(ctx);
}

void mest_context_synchronize(mest_context_t ctx) {
    try {
        static_cast<mest::Context*>(ctx)->synchronize();
    } catch (const std::exception& e) {
        set_error(e.what());
    }
}

int mest_device_count() {
    try {
        return mest::Device::count();
    } catch (...) {
        return 0;
    }
}

const char* mest_device_name(int device_id) {
    static thread_local std::string name;
    try {
        name = mest::Device::query(device_id).name;
        return name.c_str();
    } catch (const std::exception& e) {
        set_error(e.what());
        return "";
    }
}

long long mest_device_total_memory(int device_id) {
    try {
        return static_cast<long long>(mest::Device::query(device_id).total_memory);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    }
}

mest_buffer_t mest_buffer_create_f32(size_t count) {
    try {
        return new F32Buffer{mest::Buffer<float>(count), count};
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

mest_buffer_t mest_buffer_create_f64(size_t count) {
    try {
        return new F64Buffer{mest::Buffer<double>(count), count};
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

void mest_buffer_destroy(mest_buffer_t buf) {
    delete static_cast<F32Buffer*>(buf);
}

void mest_buffer_upload_f32(mest_buffer_t buf, const float* src, size_t count) {
    try {
        static_cast<F32Buffer*>(buf)->buf.upload(src, count);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_buffer_download_f32(mest_buffer_t buf, float* dst, size_t count) {
    try {
        static_cast<F32Buffer*>(buf)->buf.download(dst, count);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_buffer_upload_f64(mest_buffer_t buf, const double* src, size_t count) {
    try {
        static_cast<F64Buffer*>(buf)->buf.upload(src, count);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_buffer_download_f64(mest_buffer_t buf, double* dst, size_t count) {
    try {
        static_cast<F64Buffer*>(buf)->buf.download(dst, count);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_buffer_fill_f32(mest_buffer_t buf, float value, size_t count) {
    try {
        auto* b = static_cast<F32Buffer*>(buf);
        auto  cfg = mest::make_config(count);
        extern __global__ void fill_f32_kernel(float*, float, int);
        (void)cfg; (void)b; (void)value;
        b->buf.fill(0);
    } catch (const std::exception& e) { set_error(e.what()); }
}

size_t mest_buffer_count(mest_buffer_t buf) {
    return static_cast<F32Buffer*>(buf)->count;
}

static float* f32ptr(mest_buffer_t b) { return static_cast<F32Buffer*>(b)->buf.ptr(); }
static double* f64ptr(mest_buffer_t b) { return static_cast<F64Buffer*>(b)->buf.ptr(); }

namespace mest { namespace kernels {
    __global__ void vector_add_f32(const float*, const float*, float*, int);
    __global__ void vector_scale_f32(float*, float, int);
    __global__ void vector_add_f64(const double*, const double*, double*, int);
    __global__ void matrix_transpose_f32(const float*, float*, int, int);
}}

void mest_vector_add_f32(mest_buffer_t a, mest_buffer_t b, mest_buffer_t c, size_t n) {
    try {
        auto cfg = mest::make_config(n);
        MEST_LAUNCH(mest::kernels::vector_add_f32, cfg, f32ptr(a), f32ptr(b), f32ptr(c), (int)n);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_vector_scale_f32(mest_buffer_t data, float scalar, size_t n) {
    try {
        auto cfg = mest::make_config(n);
        MEST_LAUNCH(mest::kernels::vector_scale_f32, cfg, f32ptr(data), scalar, (int)n);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_vector_add_f64(mest_buffer_t a, mest_buffer_t b, mest_buffer_t c, size_t n) {
    try {
        auto cfg = mest::make_config(n);
        MEST_LAUNCH(mest::kernels::vector_add_f64, cfg, f64ptr(a), f64ptr(b), f64ptr(c), (int)n);
    } catch (const std::exception& e) { set_error(e.what()); }
}

void mest_matrix_transpose_f32(mest_buffer_t input, mest_buffer_t output, int rows, int cols) {
    try {
        auto cfg = mest::make_config_2d(cols, rows);
        MEST_LAUNCH(mest::kernels::matrix_transpose_f32, cfg, f32ptr(input), f32ptr(output), rows, cols);
    } catch (const std::exception& e) { set_error(e.what()); }
}

const char* mest_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
