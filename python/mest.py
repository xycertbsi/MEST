import ctypes
import ctypes.util
import os
import sys
import numpy as np
from typing import Optional, List


def _load_library():
    search_paths = [
        os.path.join(os.path.dirname(__file__), "libmest.so"),
        os.path.join(os.path.dirname(__file__), "../build/libmest.so"),
        "libmest.so",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)
    raise OSError(
        "libmest.so not found. Build MEST first:\n"
        "  cd build && cmake .. && make"
    )


_lib = _load_library()

_lib.mest_context_create.restype        = ctypes.c_void_p
_lib.mest_context_create.argtypes       = [ctypes.c_int]
_lib.mest_context_destroy.argtypes      = [ctypes.c_void_p]
_lib.mest_context_synchronize.argtypes  = [ctypes.c_void_p]
_lib.mest_device_count.restype          = ctypes.c_int
_lib.mest_device_name.restype           = ctypes.c_char_p
_lib.mest_device_name.argtypes          = [ctypes.c_int]
_lib.mest_device_total_memory.restype   = ctypes.c_longlong
_lib.mest_device_total_memory.argtypes  = [ctypes.c_int]
_lib.mest_buffer_create_f32.restype     = ctypes.c_void_p
_lib.mest_buffer_create_f32.argtypes    = [ctypes.c_size_t]
_lib.mest_buffer_create_f64.restype     = ctypes.c_void_p
_lib.mest_buffer_create_f64.argtypes    = [ctypes.c_size_t]
_lib.mest_buffer_destroy.argtypes       = [ctypes.c_void_p]
_lib.mest_buffer_upload_f32.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),  ctypes.c_size_t]
_lib.mest_buffer_download_f32.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),  ctypes.c_size_t]
_lib.mest_buffer_upload_f64.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.mest_buffer_download_f64.argtypes  = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.mest_buffer_count.restype          = ctypes.c_size_t
_lib.mest_buffer_count.argtypes         = [ctypes.c_void_p]
_lib.mest_vector_add_f32.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.mest_vector_scale_f32.argtypes     = [ctypes.c_void_p, ctypes.c_float, ctypes.c_size_t]
_lib.mest_vector_add_f64.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.mest_matrix_transpose_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.mest_last_error.restype            = ctypes.c_char_p


def _check_error():
    err = _lib.mest_last_error()
    if err:
        raise RuntimeError(f"MEST error: {err.decode()}")


class DeviceInfo:
    def __init__(self, device_id: int):
        self.id = device_id
        self.name = _lib.mest_device_name(device_id).decode()
        self.total_memory = _lib.mest_device_total_memory(device_id)

    def __repr__(self):
        mem_gb = self.total_memory / (1024 ** 3)
        return f"DeviceInfo(id={self.id}, name='{self.name}', memory={mem_gb:.1f} GB)"


class Buffer:
    def __init__(self, data: np.ndarray):
        data = np.ascontiguousarray(data)
        self._count = data.size
        self._dtype = data.dtype

        if data.dtype == np.float32:
            self._ptr = _lib.mest_buffer_create_f32(self._count)
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _lib.mest_buffer_upload_f32(self._ptr, ptr, self._count)
        elif data.dtype == np.float64:
            self._ptr = _lib.mest_buffer_create_f64(self._count)
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _lib.mest_buffer_upload_f64(self._ptr, ptr, self._count)
        else:
            raise TypeError(f"Unsupported dtype: {data.dtype}. Use float32 or float64.")

    @classmethod
    def empty_f32(cls, count: int) -> "Buffer":
        obj = object.__new__(cls)
        obj._ptr   = _lib.mest_buffer_create_f32(count)
        obj._count = count
        obj._dtype = np.float32
        return obj

    def to_numpy(self) -> np.ndarray:
        result = np.empty(self._count, dtype=self._dtype)
        if self._dtype == np.float32:
            ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _lib.mest_buffer_download_f32(self._ptr, ptr, self._count)
        else:
            ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _lib.mest_buffer_download_f64(self._ptr, ptr, self._count)
        return result

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.mest_buffer_destroy(self._ptr)

    def __len__(self):
        return self._count

    @property
    def handle(self):
        return self._ptr


class Context:
    def __init__(self, device_id: int = 0):
        self._ptr = _lib.mest_context_create(device_id)
        if not self._ptr:
            raise RuntimeError("Failed to create MEST context")

    def synchronize(self):
        _lib.mest_context_synchronize(self._ptr)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()

    def destroy(self):
        if self._ptr:
            _lib.mest_context_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.destroy()


def device_count() -> int:
    return _lib.mest_device_count()


def list_devices() -> List[DeviceInfo]:
    return [DeviceInfo(i) for i in range(device_count())]


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    assert len(a) == len(b), "Vectors must be the same length"

    buf_a = Buffer(a)
    buf_b = Buffer(b)
    buf_c = Buffer.empty_f32(len(a))

    _lib.mest_vector_add_f32(buf_a.handle, buf_b.handle, buf_c.handle, len(a))
    _check_error()

    return buf_c.to_numpy()


def vector_scale(data: np.ndarray, scalar: float) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.float32)
    buf  = Buffer(data)
    _lib.mest_vector_scale_f32(buf.handle, ctypes.c_float(scalar), len(data))
    _check_error()
    return buf.to_numpy()


def matrix_transpose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, cols = matrix.shape
    buf_in  = Buffer(matrix.flatten())
    buf_out = Buffer.empty_f32(rows * cols)
    _lib.mest_matrix_transpose_f32(buf_in.handle, buf_out.handle, rows, cols)
    _check_error()
    return buf_out.to_numpy().reshape(cols, rows)
