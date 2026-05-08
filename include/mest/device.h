#pragma once

#include "error.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace mest {

struct DeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
};

class Device {
public:
    explicit Device(int id = 0) : id_(id) {
        MEST_CHECK(cudaSetDevice(id_));
    }

    static int count() {
        int n = 0;
        MEST_CHECK(cudaGetDeviceCount(&n));
        return n;
    }

    static std::vector<DeviceInfo> list_all() {
        int n = count();
        std::vector<DeviceInfo> devices;
        devices.reserve(n);
        for (int i = 0; i < n; ++i) {
            devices.push_back(query(i));
        }
        return devices;
    }

    static DeviceInfo query(int id) {
        cudaDeviceProp prop;
        MEST_CHECK(cudaGetDeviceProperties(&prop, id));

        size_t free_mem = 0, total_mem = 0;
        int current;
        MEST_CHECK(cudaGetDevice(&current));
        MEST_CHECK(cudaSetDevice(id));
        MEST_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        MEST_CHECK(cudaSetDevice(current));

        return DeviceInfo{
            id,
            std::string(prop.name),
            total_mem,
            free_mem,
            prop.major,
            prop.minor,
            prop.multiProcessorCount,
            prop.maxThreadsPerBlock,
            prop.warpSize
        };
    }

    DeviceInfo info() const { return query(id_); }

    void synchronize() const {
        MEST_CHECK(cudaSetDevice(id_));
        MEST_CHECK(cudaDeviceSynchronize());
    }

    void reset() const {
        MEST_CHECK(cudaSetDevice(id_));
        MEST_CHECK(cudaDeviceReset());
    }

    int id() const { return id_; }

private:
    int id_;
};

} // namespace mest
