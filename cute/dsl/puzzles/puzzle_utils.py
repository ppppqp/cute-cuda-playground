"""Small cuda-python harness shared by the CuTe DSL puzzles."""

from __future__ import annotations

import numpy as np
import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cuda_runtime

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


DTYPES = {
    np.dtype(np.float16): cutlass.Float16,
    np.dtype(np.float32): cutlass.Float32,
    np.dtype(np.int8): cutlass.Int8,
}


def cuda_check(result):
    error, *values = result
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        _, message = cuda_runtime.cudaGetErrorString(error)
        text = message.decode() if isinstance(message, bytes) else str(message)
        raise RuntimeError(f"CUDA runtime error {int(error)}: {text}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class DeviceArray:
    def __init__(self, host: np.ndarray):
        self.host = np.ascontiguousarray(host)
        self.address = cuda_check(cuda_runtime.cudaMalloc(self.host.nbytes))
        cuda_check(cuda_runtime.cudaMemcpy(
            self.address, self.host.ctypes.data, self.host.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ))

    @property
    def ptr(self):
        return make_ptr(
            DTYPES[self.host.dtype], self.address, cute.AddressSpace.gmem,
            assumed_align=16,
        )

    def download(self) -> np.ndarray:
        out = np.empty_like(self.host)
        cuda_check(cuda_runtime.cudaMemcpy(
            out.ctypes.data, self.address, out.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ))
        return out

    def close(self) -> None:
        if self.address is not None:
            cuda_runtime.cudaFree(self.address)
            self.address = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def stream():
    return cuda_driver.CUstream(0)


def synchronize() -> None:
    cuda_check(cuda_runtime.cudaDeviceSynchronize())


def require_compute_capability(major: int, minor: int = 0) -> None:
    device = cuda_check(cuda_runtime.cudaGetDevice())
    properties = cuda_check(cuda_runtime.cudaGetDeviceProperties(device))
    actual = (properties.major, properties.minor)
    if actual < (major, minor):
        raise RuntimeError(
            f"this puzzle requires SM{major}{minor}; current device is SM{actual[0]}{actual[1]}"
        )


def random(shape, seed: int, lo=-1.0, hi=1.0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lo, hi, shape).astype(np.float32)


def unfinished(shape) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def check(got: np.ndarray, expected: np.ndarray, *, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)
    print(f"PASS: all {got.size} values match")
