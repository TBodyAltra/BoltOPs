from pathlib import Path
import json,inspect,importlib.metadata as metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry
r=Path('/xukang/code/pmd-triton-sessions/byte-quant-matmul-20261105/upstream/projects/micro_perf')
ProviderRegistry.load_all_vendor_impls(r/'op_defs',[r/'vendor_ops/GPU/ops'])
q=ProviderRegistry.OP_MAPPING['quant_matmul']
result=dict(loaded_provider_operators={name:list(ops) for name,ops in ProviderRegistry.OP_PROVIDER_MAPPING.items()},quant_matmul={name:dict(class_name=cls.__qualname__,module=cls.__module__,file=inspect.getfile(cls),parser_file=inspect.getfile(cls.vendor_parser),parser_source=inspect.getsource(cls.vendor_parser)) for name,cls in q.items()},installed_xpu_distributions=[dict(name=dist.metadata['Name'],version=dist.version) for dist in metadata.distributions() if 'xpu' in dist.metadata.get('Name','').lower()])
Path('registry.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
