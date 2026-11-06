set -e
source /xukang/code/pmd-triton-sessions/single-tg-pipeline-20261028-round339-persistent-wasp-wdra/env-native-v2.sh
python3 -m venv --system-site-packages /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-venv
/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-venv/bin/python -m pip install --no-index --no-deps /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/wheels/*.whl
export PYTHONPATH=/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/deps:$PYTHONPATH
/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/upstream
mkdir -p /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-record
cd /xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-record
/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-venv/bin/python -c 'import importlib.metadata as m; print("EDITABLE_VERSION",m.version("xpu-perf")); import xpu_perf.micro_perf.core.op as p; print("CORE_MODULE",p.__file__)'
/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/registry-audit/editable-venv/bin/python ../inspect_registry.py
