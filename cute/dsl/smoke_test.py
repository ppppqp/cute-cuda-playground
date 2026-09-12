"""Compile, launch, and verify a small CuTe DSL kernel from Python.

This uses cuda-python rather than Torch to keep the smoke test lightweight. The
kernel computes y[i] = 2*x[i] + 1.
"""

from importlib.metadata import version

import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cuda_runtime
import numpy as np

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


THREADS = 256


@cute.kernel
def saxpy_smoke_kernel(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int32):
    """One-dimensional SIMT kernel: y = 2*x + 1."""
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    index = block * THREADS + thread
    if index < n:
        y[index] = 2.0 * x[index] + 1.0


@cute.jit
def launch_saxpy_smoke(
    x_ptr: cute.Pointer,
    y_ptr: cute.Pointer,
    n: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    """Construct CuTe tensors and launch the device kernel."""
    layout = cute.make_layout(n)
    x = cute.make_tensor(x_ptr, layout)
    y = cute.make_tensor(y_ptr, layout)
    saxpy_smoke_kernel(x, y, n).launch(
        grid=(cute.ceil_div(n, THREADS), 1, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


def cuda_check(result):
    """Unwrap a cuda-python runtime result tuple or raise a readable error."""
    error, *values = result
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        _, message = cuda_runtime.cudaGetErrorString(error)
        text = message.decode() if isinstance(message, bytes) else str(message)
        raise RuntimeError(f"CUDA runtime error {int(error)}: {text}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def main() -> None:
    installed = version("nvidia-cutlass-dsl")
    assert installed == "4.6.1", f"expected CuTe DSL 4.6.1, found {installed}"

    n = 1003  # deliberately not divisible by the thread-block size
    host_x = np.linspace(-3.0, 4.0, n, dtype=np.float32)
    host_y = np.empty_like(host_x)
    nbytes = host_x.nbytes
    device_x = None
    device_y = None

    try:
        device_x = cuda_check(cuda_runtime.cudaMalloc(nbytes))
        device_y = cuda_check(cuda_runtime.cudaMalloc(nbytes))
        cuda_check(cuda_runtime.cudaMemcpy(
            device_x, host_x.ctypes.data, nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ))

        x_ptr = make_ptr(
            cutlass.Float32, device_x, cute.AddressSpace.gmem, assumed_align=16
        )
        y_ptr = make_ptr(
            cutlass.Float32, device_y, cute.AddressSpace.gmem, assumed_align=16
        )
        launch_saxpy_smoke(x_ptr, y_ptr, n, cuda_driver.CUstream(0))
        cuda_check(cuda_runtime.cudaDeviceSynchronize())
        cuda_check(cuda_runtime.cudaMemcpy(
            host_y.ctypes.data, device_y, nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        ))
    finally:
        if device_y is not None:
            cuda_runtime.cudaFree(device_y)
        if device_x is not None:
            cuda_runtime.cudaFree(device_x)

    expected = 2.0 * host_x + 1.0
    np.testing.assert_allclose(host_y, expected, rtol=1.0e-6, atol=1.0e-6)
    print(f"PASS: CuTe DSL {installed} JIT kernel produced {n} correct values")


if __name__ == "__main__":
    main()
