#!/bin/bash
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round339-persistent-wasp-wdra/env-native-v2.sh
export PYTHONPATH=/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/deps:/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/smoke/src:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GPU_ARGS="['--abs-max-tick=100000000000']"
unset HSA_TOOLS_LIB
cd /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/smoke-int8
python3 -u ../smoke/projects/micro_perf/launch.py --backend GPU --workload ../workloads/int8.json --numa=-1 --device 0 --report_dir reports
result_code=$?
printf '%s
' "$result_code" > exit-code
exit "$result_code"
