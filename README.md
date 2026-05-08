# MEST — CUDA Made Easy

**MEST** is a lightweight C++ library that wraps NVIDIA CUDA into a clean, modern API. It removes the boilerplate from GPU programming and exposes its functionality to both Python and Java through thin bindings.

---

## Features

- **Device management** — query GPUs, count devices, read memory stats
- **Smart GPU buffers** — RAII-managed device memory with easy upload/download
- **Pinned host memory** — for fast async transfers
- **Kernel launcher** — clean `make_config` + `MEST_LAUNCH` macro
- **Stream management** — async execution with callbacks
- **Context object** — single entry point that owns a device + stream
- **C API** — a stable C bridge for FFI from any language
- **Python bindings** — via `ctypes`, zero extra dependencies
- **Java bindings** — via JNI, with idiomatic `AutoCloseable` wrappers

---

## Requirements

| Component       | Minimum version |
|----------------|-----------------|
| CUDA Toolkit   | 11.0+           |
| CMake          | 3.18+           |
| C++ compiler   | GCC 9+ / MSVC 2019+ |
| Python (opt.)  | 3.8+ + NumPy    |
| Java (opt.)    | JDK 11+         |

---

## Building

```bash
git clone https://github.com/your-org/MEST
cd MEST
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMEST_BUILD_PYTHON=ON \
  -DMEST_BUILD_JAVA=ON \
  -DMEST_BUILD_EXAMPLES=ON \
  -DMEST_BUILD_TESTS=ON

make -j$(nproc)
```

After building, `libmest.so` (and `libmest_jni.so` for Java) will be in `build/`.

To install system-wide:

```bash
sudo make install
```

---

## C++ Usage

### 1. Include the master header

```cpp
#include <mest/mest.h>
```

### 2. List devices

```cpp
auto devices = mest::Device::list_all();
for (const auto& d : devices) {
    std::cout << d.name << "  " << d.total_memory / 1e9 << " GB\n";
}
```

### 3. Allocate and transfer GPU memory

```cpp
std::vector<float> host_data = {1.f, 2.f, 3.f, 4.f};

auto gpu_buf = mest::upload(host_data);         // host → GPU
auto result  = gpu_buf.download();              // GPU → host

auto empty   = mest::make_buffer<float>(1024); // uninitialized GPU buffer
empty.fill(0);                                  // zero the memory
```

### 4. Launch a kernel

```cpp
__global__ void my_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.f;
}

auto buf = mest::upload(host_data);
auto cfg = mest::make_config(host_data.size()); // auto grid/block
MEST_LAUNCH(my_kernel, cfg, buf.ptr(), (int)host_data.size());
```

For 2D kernels:

```cpp
auto cfg2d = mest::make_config_2d(width, height, 16, 16);
MEST_LAUNCH(my_2d_kernel, cfg2d, buf.ptr(), width, height);
```

### 5. Use streams

```cpp
mest::Stream stream = mest::Stream::non_blocking();
auto cfg = mest::make_config(N, 256, 0, &stream);
MEST_LAUNCH(my_kernel, cfg, ptr, N);
stream.synchronize();
```

### 6. Use the Context object

```cpp
mest::Context ctx(0);           // device 0
auto info = ctx.info();         // DeviceInfo
ctx.synchronize();              // sync everything
```

### 7. Error handling

All MEST functions throw `mest::CudaError` on failure, which inherits from `std::runtime_error`.

```cpp
try {
    auto buf = mest::make_buffer<float>(1'000'000'000); // may OOM
} catch (const mest::CudaError& e) {
    std::cerr << e.what() << "\n";
    std::cerr << "CUDA error code: " << e.code() << "\n";
}
```

---

## Python Usage

Copy `python/mest.py` to your project (or add it to your `PYTHONPATH`). Make sure `libmest.so` is findable.

```python
import numpy as np
import mest

# List GPUs
for d in mest.list_devices():
    print(d)

# Vector addition (GPU-accelerated)
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
result = mest.vector_add(a, b)
# result: [5.0, 7.0, 9.0]

# Scale a vector
scaled = mest.vector_scale(a.copy(), 2.0)
# scaled: [2.0, 4.0, 6.0]

# Matrix transpose
m = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
t = mest.matrix_transpose(m)
# t.shape: (3, 2)
```

### Using raw buffers in Python

```python
import ctypes
from mest import Buffer, Context

ctx = Context(0)

data = np.ones(1024, dtype=np.float32)
buf  = Buffer(data)            # uploads immediately

arr = buf.to_numpy()           # downloads back
```

---

## Java Usage

Add `libmest.so` and `libmest_jni.so` to your `java.library.path`. Include the `mest` package on your classpath.

```java
import mest.MEST;

// List GPUs
for (String d : MEST.listDevices()) {
    System.out.println(d);
}

// Vector addition
float[] a = {1f, 2f, 3f};
float[] b = {4f, 5f, 6f};
float[] result = MEST.vectorAdd(a, b);
// result: [5.0, 7.0, 9.0]

// Scale
float[] scaled = MEST.vectorScale(a.clone(), 2.0f);

// Matrix transpose
float[][] matrix    = {{1f, 2f, 3f}, {4f, 5f, 6f}};
float[][] transposed = MEST.matrixTranspose(matrix);
```

### Low-level Java buffer control

```java
import mest.Buffer;
import mest.Context;

try (Context ctx = new Context(0);
     Buffer buf  = Buffer.ofFloats(new float[]{1, 2, 3, 4})) {

    float[] back = buf.toFloatArray();
}   // ctx and buf auto-closed
```

---

## Project Structure

```
MEST/
├── include/mest/
│   ├── mest.h          ← master include
│   ├── device.h        ← device query & management
│   ├── memory.h        ← Buffer<T> and PinnedBuffer<T>
│   ├── kernel.h        ← make_config, MEST_LAUNCH
│   ├── stream.h        ← Stream (RAII)
│   ├── context.h       ← Context (device + stream)
│   ├── error.h         ← CudaError, MEST_CHECK
│   ├── capi.h          ← C API for FFI
│   └── jni_bridge.h    ← JNI declarations
├── src/
│   ├── kernels.cu      ← built-in GPU kernels
│   ├── capi.cu         ← C API implementation
│   └── jni_bridge.cu   ← JNI bridge implementation
├── python/
│   └── mest.py         ← Python bindings (ctypes)
├── java/src/main/java/mest/
│   ├── MEST.java       ← high-level Java API
│   ├── Buffer.java     ← AutoCloseable GPU buffer
│   ├── Context.java    ← AutoCloseable context
│   └── NativeBridge.java ← raw JNI declarations
├── examples/
│   ├── cpp/            ← hello.cu, vector_ops.cu
│   ├── python/         ← example.py
│   └── java/           ← Example.java
├── tests/
│   └── test_mest.cu    ← unit tests
└── CMakeLists.txt
```

---

## Running Tests

```bash
cd build
ctest --output-on-failure
# or directly:
./test_mest
```

---

## Running Examples

```bash
./mest_hello
./mest_vector_ops

python3 ../examples/python/example.py

javac -cp ../java/src/main/java ../examples/java/Example.java
java -Djava.library.path=. Example
```

---

## Extending MEST

To add your own kernel:

1. Write the `__global__` function in any `.cu` file.
2. Call `mest::make_config(N)` to get a `LaunchConfig`.
3. Use `MEST_LAUNCH(my_kernel, cfg, args...)` to launch.
4. Optionally expose via `capi.h` for Python/Java.

No registration or framework required.

---

## License

MIT License. See `LICENSE` for details.
