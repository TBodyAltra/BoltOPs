#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round313-square-wdra-fragment-confirm/build
export HCU_PIPELINE_ENV_REV=313squarewdrafragmentnative
