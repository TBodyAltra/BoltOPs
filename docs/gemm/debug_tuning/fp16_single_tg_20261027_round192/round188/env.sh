#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round188-mls-rebase-cpu/build
export HCU_PIPELINE_ENV_REV=188mlsrebasecpu
