#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round181-sgpr-liveness-native/build
export HCU_PIPELINE_ENV_REV=181sgprlivenessnative
