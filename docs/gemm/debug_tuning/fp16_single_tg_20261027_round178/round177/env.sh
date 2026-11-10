#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round177-group1-index-confirm
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round176-group1-index-native/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round176-group1-index-native/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round177-group1-index-confirm/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round177-group1-index-confirm/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round177-group1-index-confirm/build
export HCU_PIPELINE_ENV_REV=177group1indexconfirm
