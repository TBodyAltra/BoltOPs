"""Decode OCP E8M0 scales for FP8 dot-then-scale kernels."""

import triton
import triton.language as tl


@triton.jit
def decode_fp8_scale(scale, FP8_INPUT: tl.constexpr):
    """Treat uint8 scale storage as E8M0 only for FP8 operands.

    E8M0 has exponent bias 127, no zero encoding, and reserves 255 for NaN.
    Building the FP32 bits is exact, including code 0 (2**-127), and avoids
    exp2 underflow/approximation. Floating scales and non-FP8 paths pass
    through unchanged.
    """
    if FP8_INPUT and scale.dtype == tl.uint8:
        exponent = scale.to(tl.uint32)
        bits = exponent << 23
        bits = tl.where(exponent == 0, 0x00400000, bits)
        bits = tl.where(exponent == 255, 0x7FC00000, bits)
        return bits.to(tl.float32, bitcast=True)
    else:
        return scale
