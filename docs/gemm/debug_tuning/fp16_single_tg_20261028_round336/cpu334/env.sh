#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round334-unroll-stage1-barrier-cpu/build
export HCU_PIPELINE_ENV_REV=334squarewdra4fragmentcpu
