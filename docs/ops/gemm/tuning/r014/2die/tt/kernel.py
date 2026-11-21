import triton

import triton.language as tl

@triton.jit
def gemm_row_reduce(a,b,bias,c,M,N,K,stride_am,stride_ak,stride_bk,stride_bn,stride_ck,stride_cm,stride_cn,
                    BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,BLOCK_SIZE_K:tl.constexpr,
                    GROUP_SIZE_M:tl.constexpr,NUM_KSPLIT:tl.constexpr,SPLITK_BLOCK_SIZE:tl.constexpr,
                    cache_modifier:tl.constexpr,activation:tl.constexpr,use_activation:tl.constexpr,
                    ADD_BIAS:tl.constexpr,SKIP_REDUCE:tl.constexpr,REDUCE_K:tl.constexpr):
    row=tl.program_id(0)//tl.cdiv(N,BLOCK_SIZE_N)
    col=tl.program_id(0)%tl.cdiv(N,BLOCK_SIZE_N)*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    kk=tl.arange(0,REDUCE_K)
    av=tl.load(a+row*stride_am+kk*stride_ak,kk<K,other=0).to(tl.float32)
    bv=tl.load(b+kk[:,None]*stride_bk+col[None,:]*stride_bn,(kk[:,None]<K)&(col[None,:]<N),other=0).to(tl.float32)
    result=tl.sum(av[:,None]*bv,axis=0)
    tl.store(c+row*stride_cm+col*stride_cn,result,col<N)

