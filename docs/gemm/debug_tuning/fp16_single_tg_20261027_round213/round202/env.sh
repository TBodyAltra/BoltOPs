#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round202-fragment-bpp-native/build
export HCU_PIPELINE_ENV_REV=202fragmentbppnative
