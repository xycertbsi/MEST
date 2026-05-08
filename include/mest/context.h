#pragma once

#include "device.h"
#include "stream.h"
#include "error.h"
#include <memory>

namespace mest {

class Context {
public:
    explicit Context(int device_id = 0)
        : device_(device_id),
          default_stream_(std::make_unique<Stream>())
    {}

    Device&       device()         { return device_; }
    const Device& device()   const { return device_; }
    Stream&       stream()         { return *default_stream_; }
    const Stream& stream()   const { return *default_stream_; }

    void synchronize() { device_.synchronize(); }

    DeviceInfo info() const { return device_.info(); }

    static Context create(int device_id = 0) {
        return Context(device_id);
    }

private:
    Device              device_;
    std::unique_ptr<Stream> default_stream_;
};

} // namespace mest
