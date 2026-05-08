#pragma once

#include "error.h"
#include "stream.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace mest {

struct LaunchConfig {
    dim3         grid;
    dim3         block;
    size_t       shared_mem = 0;
    cudaStream_t stream     = nullptr;
};

inline LaunchConfig make_config(size_t total_threads,
                                 size_t block_size  = 256,
                                 size_t shared_mem  = 0,
                                 const Stream* stream = nullptr)
{
    LaunchConfig cfg;
    cfg.block      = dim3(static_cast<unsigned int>(block_size));
    cfg.grid       = dim3(static_cast<unsigned int>((total_threads + block_size - 1) / block_size));
    cfg.shared_mem = shared_mem;
    cfg.stream     = stream ? stream->handle() : nullptr;
    return cfg;
}

inline LaunchConfig make_config_2d(size_t width, size_t height,
                                    size_t bx = 16, size_t by = 16,
                                    const Stream* stream = nullptr)
{
    LaunchConfig cfg;
    cfg.block  = dim3(static_cast<unsigned int>(bx), static_cast<unsigned int>(by));
    cfg.grid   = dim3(
        static_cast<unsigned int>((width  + bx - 1) / bx),
        static_cast<unsigned int>((height + by - 1) / by)
    );
    cfg.stream = stream ? stream->handle() : nullptr;
    return cfg;
}

} // namespace mest

#define MEST_LAUNCH(kernel, cfg, ...) \
    do { \
        kernel<<<(cfg).grid, (cfg).block, (cfg).shared_mem, (cfg).stream>>>(__VA_ARGS__); \
        MEST_CHECK(cudaGetLastError()); \
    } while(0)
