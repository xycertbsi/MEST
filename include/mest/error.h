#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace mest {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t code, const std::string& context = "")
        : std::runtime_error(build_message(code, context)), code_(code) {}

    cudaError_t code() const { return code_; }

private:
    cudaError_t code_;

    static std::string build_message(cudaError_t code, const std::string& context) {
        std::string msg = "CUDA error: ";
        msg += cudaGetErrorString(code);
        if (!context.empty()) {
            msg += " [" + context + "]";
        }
        return msg;
    }
};

inline void check(cudaError_t code, const std::string& context = "") {
    if (code != cudaSuccess) {
        throw CudaError(code, context);
    }
}

#define MEST_CHECK(expr) ::mest::check((expr), #expr)

} // namespace mest
