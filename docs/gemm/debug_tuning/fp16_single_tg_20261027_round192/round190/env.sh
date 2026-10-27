#!/usr/bin/env bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261021/env.sh
export FP16_MAP_ROOT=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry
export PYTHONPATH=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry/triton/python:/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry/bolt_ops
export TRITON_HOME=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry/cache/home
export TRITON_CACHE_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry/cache/kernels
export TRITON_BUILD_DIR=/xukang/code/pmd-triton-sessions/single-tg-pipeline-20261027-round190-mls-rebase-native-retry/build
export HCU_PIPELINE_ENV_REV=190mlsrebasenative
