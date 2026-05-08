#pragma once

#include "error.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <cstring>

namespace mest {

template<typename T>
class Buffer {
public:
    explicit Buffer(size_t count) : count_(count) {
        MEST_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_   = nullptr;
        other.count_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_         = other.ptr_;
            count_       = other.count_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~Buffer() { free(); }

    void upload(const T* host_src, size_t count = 0) {
        size_t n = (count == 0) ? count_ : count;
        MEST_CHECK(cudaMemcpy(ptr_, host_src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void upload(const std::vector<T>& src) {
        upload(src.data(), src.size());
    }

    void download(T* host_dst, size_t count = 0) const {
        size_t n = (count == 0) ? count_ : count;
        MEST_CHECK(cudaMemcpy(host_dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::vector<T> download() const {
        std::vector<T> result(count_);
        download(result.data(), count_);
        return result;
    }

    void fill(int value = 0) {
        MEST_CHECK(cudaMemset(ptr_, value, count_ * sizeof(T)));
    }

    void copy_from(const Buffer<T>& other, size_t count = 0) {
        size_t n = (count == 0) ? std::min(count_, other.count_) : count;
        MEST_CHECK(cudaMemcpy(ptr_, other.ptr_, n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    T*     ptr()   const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;

    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
};

template<typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) : count_(count) {
        MEST_CHECK(cudaMallocHost(&ptr_, count_ * sizeof(T)));
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    T*     ptr()   const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

template<typename T>
Buffer<T> make_buffer(size_t count) {
    return Buffer<T>(count);
}

template<typename T>
Buffer<T> upload(const std::vector<T>& data) {
    Buffer<T> buf(data.size());
    buf.upload(data);
    return buf;
}

} // namespace mest
