#pragma once

#include "error.h"
#include <cuda_runtime.h>
#include <functional>

namespace mest {

class Stream {
public:
    Stream() {
        MEST_CHECK(cudaStreamCreate(&stream_));
    }

    explicit Stream(unsigned int flags) {
        MEST_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    ~Stream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    void synchronize() const {
        MEST_CHECK(cudaStreamSynchronize(stream_));
    }

    bool is_ready() const {
        cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaSuccess)   return true;
        if (status == cudaErrorNotReady) return false;
        MEST_CHECK(status);
        return false;
    }

    void add_callback(std::function<void()> fn) {
        auto* heap_fn = new std::function<void()>(std::move(fn));
        MEST_CHECK(cudaStreamAddCallback(
            stream_,
            [](cudaStream_t, cudaError_t, void* data) {
                auto* f = static_cast<std::function<void()>*>(data);
                (*f)();
                delete f;
            },
            heap_fn,
            0
        ));
    }

    cudaStream_t handle() const { return stream_; }

    static Stream non_blocking() {
        return Stream(cudaStreamNonBlocking);
    }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace mest
