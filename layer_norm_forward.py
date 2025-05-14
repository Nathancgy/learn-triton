import torch

import triton
import triton.language as tl

try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, stride, Mean, Rstd, N, eps, BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(axis = 0)
    X += row * stride
    Y += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for col in range(0, N, BLOCK_SIZE):
        off = col + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + off, mask = off < N, other = 0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis = 0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
    for col in range(0, N, BLOCK_SIZE):
        off = col + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + off, mask = off < N, other = 0.).to(tl.float32)
        x = tl.where(col < N, x - mean, others = 0.)
        _var += x * x
    var = tl.sum(_var, axis = 0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for col in range(0, N, BLOCK_SIZE):
        off = col + tl.arange(0, BLOCK_SIZE)
        mask = off < N
        w = tl.load(W + off, mask = mask)
        b = tl.load(B + off, mask = mask)
        x = tl.load(X + off, mask = mask, other = 0.).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + off, y, mask = mask)
        # Write output