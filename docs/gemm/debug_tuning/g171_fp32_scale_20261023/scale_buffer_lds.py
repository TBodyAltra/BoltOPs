"""Native FP32 scale copies for this aligned G171 WDRA48 experiment only."""
from pathlib import Path
import re,json,hashlib
from matrix_prefetch import normalize_mls

def transform(text):
    text,aliases=normalize_mls(text)
    start=text.index('      partition0(');end=text.index('      partition1(')
    producer=text[start:end]
    assert '#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>' in text
    old='tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>'
    count=producer.count(old);assert count>=5,count
    producer=producer.replace(old,'tensor<128xi32, #blocked>')
    layout_producer=producer.replace('tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>','tensor<128xf32, #blocked>')
    layout=text[:start]+layout_producer+text[end:]
    proofs=[]
    for label,size in [('a',128),('b',256)]:
        pattern=rf'^(\s+)(%{label}_scale_\d+) = amdg.buffer_load (%{label}_scale_ptr_\d+)\[(%{label}_scale_ptrs_\d+)\] : tensor<{size}xf32, (.+)> loc\([^\n]*\)\n'
        matches=list(re.finditer(pattern,producer,re.M));assert len(matches)==1,(label,len(matches))
        m=matches[0];indent,value,base,offset,encoding=m.groups();assert len(re.findall(re.escape(value)+r'\b',producer))==2,(label,value)
        storepat=rf'^\s+ttg.local_store {re.escape(value)}, (%\w+) : tensor<{size}xf32, .+> -> !ttg.memdesc<({size}xf32, [^\n]+)> loc\([^\n]*\)\n'
        stores=list(re.finditer(storepat,producer,re.M));assert len(stores)==1,label
        dest,desc=stores[0].groups();assert encoding in ['#blocked','#ttg.slice<{dim = 1, parent = #mma}>'],encoding
        copy=f'{indent}%scale_copy_{label} = amdg.buffer_load_to_local {base}[{offset}] into {dest} {{contiguity = 1 : i32}} : <f32>[tensor<{size}xi32, #blocked>] -> <{desc}>\n'
        wait=f'{indent}%scale_commit_{label} = ttg.async_commit_group tokens %scale_copy_{label}\n{indent}%scale_wait_{label} = ttg.async_wait %scale_commit_{label} {{num = 0 : i32}}\n'
        producer=re.sub(storepat,lambda _:wait,producer,flags=re.M)
        producer=re.sub(pattern,lambda _:copy,producer,flags=re.M)
        proofs.append({'operand':label.upper()+' scale','payload_dtype':'f32','logical_tile_elements':size,'global_indexing':'unchanged','global_B_shape':[2,48],'producer_layout':'blocked sizePerThread1 / 64 threads / 4 warps','native_operation':'amdg.buffer_load_to_local','publication':'async commit + wait0 before existing ready abarrier arrive','consumer_math_and_LDS_layout':'unchanged'})
    result=text[:start]+producer+text[end:]
    assert result[result.index('      partition1('):]==text[end:]
    assert producer.count('amdg.buffer_load_to_local ')==2 and 'amdg.buffer_load ' not in producer
    return layout,result,{'MLS_encoding_aliases':aliases,'producer_i32_layout_changes':count,'copies':proofs,'B_scale_note':'Input stays compact [2,48]; this variant preserves the existing logical 256-entry B scale LDS staging and repeated source indices.'}

if __name__=='__main__':
    import sys
    src=Path(sys.argv[1]);root=Path(sys.argv[2]);original=src.read_text();normalized,_=normalize_mls(original);layout,changed,proof=transform(original)
    for name,data in [('mls-scale-IR-control.ttgir',normalized),('mls-scale-layout-control.ttgir',layout),('mls-scale-buffer-LDS.ttgir',changed)]:
        (root/name).write_text(data)
    proof['source']=str(src);proof['source_sha256']=hashlib.sha256(original.encode()).hexdigest();proof['executed_candidate_sha256']=hashlib.sha256(changed.encode()).hexdigest()
    (root/'scale-buffer-LDS-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Prepared native FP32 scale copies; A/B MLS, physical scale layout and consumer mathematics unchanged')
