"""R016 TT-only specialization: BN32 / 1 wave / local-prefetch, M/N/K and strides are runtime arguments.
Needs no compiler change and no process-local backend option.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M','N','K'])
def _gemm_bf16_generic_kernel(
    a_ptr,b_ptr,c_ptr,bias_ptr,
    M,N,K,
    stride_am,stride_ak,
    stride_bk,stride_bn,
    stride_ck,stride_cm,stride_cn,
    BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,BLOCK_SIZE_K:tl.constexpr,
    GROUP_SIZE_M:tl.constexpr,NUM_KSPLIT:tl.constexpr,SPLITK_BLOCK_SIZE:tl.constexpr,
    WARP_SPECIALIZE:tl.constexpr,USE_MLS:tl.constexpr,USE_MATRIX_STORE:tl.constexpr,
    LOOP_UNROLL_FACTOR:tl.constexpr=1,
):
    tl.static_assert(BLOCK_SIZE_M==16 and NUM_KSPLIT==1)
    tl.static_assert(GROUP_SIZE_M==1)
    tl.static_assert(USE_MLS and not USE_MATRIX_STORE)
    tl.assume(M>0)
    tl.assume(N>0)
    tl.assume(stride_am>0)
    tl.assume(stride_ak>0)
    tl.assume(stride_bk>0)
    tl.assume(stride_bn>0)
    tl.assume(stride_cm>0)
    tl.assume(stride_cn>0)
    tl.assume(K>0)
    tl.assume(K%BLOCK_SIZE_K==0)
    on=tl.program_id(0)*BLOCK_SIZE_N
    acc=tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),tl.float32)
    for i in tl.range(0,tl.cdiv(K,BLOCK_SIZE_K),
                      warp_specialize=WARP_SPECIALIZE,loop_unroll_factor=LOOP_UNROLL_FACTOR):
        ok=i*BLOCK_SIZE_K
        a=tl.matrix_load(a_ptr,shape=[M,K],strides=[stride_am,stride_ak],
                         block_shape=[BLOCK_SIZE_M,BLOCK_SIZE_K],offsets=[0,ok],boundary_check=(0,))
        b=tl.matrix_load(b_ptr+on,shape=[K,BLOCK_SIZE_N],strides=[stride_bk,stride_bn],
                         block_shape=[BLOCK_SIZE_K,BLOCK_SIZE_N],offsets=[ok,0])
        acc=tl.dot(a,b,acc)
    rm=tl.arange(0,BLOCK_SIZE_M)
    rn=on+tl.arange(0,BLOCK_SIZE_N)
    tl.store(c_ptr+rm[:,None]*stride_cm+rn[None,:],acc.to(c_ptr.dtype.element_ty),mask=rm[:,None]<M)


def gemm_r016(a,b,out=None):
    """Full-shape checked entry; A[12,2048] @ B[2048,9216], BF16/FP32/BF16.
    Only row-major TT and no bias/activation/split-K.
    """
    import torch
    if a.shape!=(12,2048) or b.shape!=(2048,9216):
        raise ValueError('requires A[12,2048], B[2048,9216]')
    if a.dtype!=torch.bfloat16 or b.dtype!=torch.bfloat16:
        raise ValueError('requires BF16 inputs')
    if not a.is_cuda or b.device!=a.device:
        raise ValueError('inputs must share a gfx946 device')
    if a.stride(1)!=1 or b.stride(1)!=1 or a.stride(0)<2048 or b.stride(0)<9216:
        raise ValueError('requires non-overlapping row-major TT input strides')
    from triton.runtime import driver
    if driver.active.get_current_target().arch!='gfx946':
        raise ValueError('this specialization has been validated only on gfx946')
    if out is None:
        out=torch.empty((12,9216),dtype=torch.bfloat16,device=a.device)
    if (out.shape!=(12,9216) or out.dtype!=torch.bfloat16 or out.device!=a.device or not out.is_contiguous()):
        raise ValueError('out must be contiguous BF16 [12,9216] on input device')
    out_lo=out.data_ptr()
    out_hi=out_lo+out.numel()*out.element_size()
    for operand in (a,b):
        lo=operand.data_ptr()
        hi=lo+((operand.shape[0]-1)*operand.stride(0)+operand.shape[1])*operand.element_size()
        if out_lo<hi and lo<out_hi:
            raise ValueError('output must not overlap either input')
    _gemm_bf16_generic_kernel[(288,)](
        a,b,out,a,12,9216,2048,*a.stride(),*b.stride(),0,*out.stride(),
        BLOCK_SIZE_M=16,BLOCK_SIZE_N=32,BLOCK_SIZE_K=64,GROUP_SIZE_M=1,
        NUM_KSPLIT=1,SPLITK_BLOCK_SIZE=0,WARP_SPECIALIZE=False,USE_MLS=True,
        USE_MATRIX_STORE=False,LOOP_UNROLL_FACTOR=1,
        num_warps=1,num_stages=2,mmac_layout_force=3,schedule_hint='local-prefetch')
    return out
