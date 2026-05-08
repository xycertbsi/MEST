#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* mest_context_t;
typedef void* mest_buffer_t;
typedef void* mest_stream_t;

mest_context_t mest_context_create(int device_id);
void           mest_context_destroy(mest_context_t ctx);
void           mest_context_synchronize(mest_context_t ctx);
int            mest_device_count();
const char*    mest_device_name(int device_id);
long long      mest_device_total_memory(int device_id);

mest_buffer_t  mest_buffer_create_f32(size_t count);
mest_buffer_t  mest_buffer_create_f64(size_t count);
void           mest_buffer_destroy(mest_buffer_t buf);
void           mest_buffer_upload_f32(mest_buffer_t buf, const float* src, size_t count);
void           mest_buffer_download_f32(mest_buffer_t buf, float* dst, size_t count);
void           mest_buffer_upload_f64(mest_buffer_t buf, const double* src, size_t count);
void           mest_buffer_download_f64(mest_buffer_t buf, double* dst, size_t count);
void           mest_buffer_fill_f32(mest_buffer_t buf, float value, size_t count);
size_t         mest_buffer_count(mest_buffer_t buf);

void mest_vector_add_f32(mest_buffer_t a, mest_buffer_t b, mest_buffer_t c, size_t n);
void mest_vector_scale_f32(mest_buffer_t data, float scalar, size_t n);
void mest_vector_add_f64(mest_buffer_t a, mest_buffer_t b, mest_buffer_t c, size_t n);
void mest_matrix_transpose_f32(mest_buffer_t input, mest_buffer_t output, int rows, int cols);

const char* mest_last_error();

#ifdef __cplusplus
}
#endif
