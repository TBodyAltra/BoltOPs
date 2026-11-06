"""Scale-storage decoding for Gluon kernels; no quantization or MMAC emulation."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def decode_scale(value, E8M0: gl.constexpr):
    # Same E8M0 value contract as boltops.utility.fp8_scale.decode_fp8_scale.
    if E8M0:
        exponent = value.to(gl.uint32)
        bits = exponent << 23
        bits = gl.where(exponent == 0, 0x00400000, bits)
        bits = gl.where(exponent == 255, 0x7FC00000, bits)
        return bits.to(gl.float32, bitcast=True)
    else:
        return value
