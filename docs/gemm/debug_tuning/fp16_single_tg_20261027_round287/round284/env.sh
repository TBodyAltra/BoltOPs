#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round284-64square-vector-stable-native/build
export HCU_PIPELINE_ENV_REV=28464squarevectorstablenative
