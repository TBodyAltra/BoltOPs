"""Insert January dense GEMM requirements from the frozen row 72/86 Excel extract."""
from pathlib import Path
import csv,json,re

AUDIT=Path(__file__).resolve().parent.parent/'byte_january_20261105'


def integrate(text, projected, stats):
    source=json.loads((AUDIT/'source.json').read_text())
    policy=json.loads((AUDIT/'implementation-policy.json').read_text())
    assert source['included_rows']==[72,86]
    def renumber(match):
        value=int(match[2])
        return match[1]+f'{value+40 if value>=47 else value:03}'
    text=re.sub(r'([Rr])(\d{3})(?!\d)',renumber,text)
    for r in projected:
        for field in ['requirement_id','anchor']:
            r[field]=re.sub(r'([Rr])(\d{3})',renumber,r[field])
    repo=policy['hygon_repository']+'/-/blob/'+policy['hygon_commit']+'/projects/micro_perf/'
    def link(label,path):return f'[{label}]({repo+path})'
    section='<a name="january-byte-p1"></a>\n\n## 1 月字节送测需求（P1）\n\n'
    section+='需求来源：`GPU单体测试用例 v2.1.xlsx` → `测试用例` 第 72–97 行，纳入第 72、86 行；MoE 及非 GEMM 用例不纳入。以下为通用算子测试，不限定模型。\n\n'
    section+='补充约定：INT8 默认输出 BF16；未规定的量化 scale 使用 FP32，A 按 token、B 按输出通道。MXFP8/MXFP4 保留标准 E8M0 / 32 元素分组；Basic BF16 不使用量化 scale。\n\n'
    section+='验收要求：性能与厂商宣称值的差异不超过 10%，测试过程无软硬件异常；宣称值及对应的精度、频率和测量口径需随送测结果提供。\n\n'
    new=[];expanded=[];summaries=[];number=46
    for kind,title,excelrow,op,M,variants,kn in [
        ('byte-basic','GEMM Basic（GPU_Micro_001）',72,'gemm',source['basic']['M'],source['basic']['dtypes'],source['basic']['KN']),
        ('byte-quant','quant_matmul（GPU_Micro_015）',86,'quant_matmul',source['quant']['num_tokens'],source['quant']['dtypes'],[(x[1],x[2]) for x in source['quant']['sp_K_N']])]:
        first=number+1
        section+=f'<a name="model-{kind}"></a>\n\n### {title}\n\n'
        section+=f'需求来源：`GPU单体测试用例 v2.1.xlsx`，`测试用例!B{excelrow}:E{excelrow}`。\n\n'
        mlabel='M₇₂' if excelrow==72 else 'M₈₆'
        section+=f'`{mlabel}` = `'+', '.join(map(str,M))+'`。每行覆盖此列表全部 M；矩阵维度统一写为 **M × N × K**。\n\n'
        if excelrow==72:
            section+='Excel 的 KN 按 **K、N** 解释：A[M,K] × B[K,N] → C[M,N]；FP4 为“如有”条件项。输出每组 shape 的算力。本文 BF16、INT8、FP8 默认输出 BF16；Basic FP4 的输出类型仍未规定。精度与 scale 采用下表口径，不表示相应 GPU 实现已验收。\n\n'
        else:
            section+='`sp_size=1`，M=`num_tokens`、K=`hidden_size`、N=`new_hidden_size`。表内精度顺序为 **输入 / 权重 / 计算 / 输出**，scale 约定及运行结果见下方核对。\n\n'
            section+='FP4、MXFP4 均需覆盖；输出 latency、算力、带宽、MFU、MBU。\n\n'
        section+='| 编号 | 算子 / 精度 | 层 / 算子名 | M × N × K | batch / E | 阶段 | 来源数 | A scale | B scale | 接口参考 / 状态 |\n'
        section+='| '+' | '.join(['---']*10)+' |\n'
        for dtype in variants:
            for K,N in kn:
                number+=1;rid=f'R{number:03}';target=rid.lower()
                precision=dtype if isinstance(dtype,str) else ' / '.join(dtype)
                outdtype=policy['basic_output_dtype'][dtype] if excelrow==72 else dtype[-1]
                element=dtype if excelrow==72 else dtype[0]
                display_precision=(f'{dtype} × {dtype} → {outdtype}' if excelrow==72 else precision)
                optional=excelrow==72 and dtype=='float4'
                if excelrow==72 and element=='bfloat16':
                    aa=bb='—'
                    scale_a_dtype=scale_b_dtype=scale_a_grain=scale_b_grain='—'
                elif element.startswith('mx'):
                    aa=bb='E8M0 / 32 元素（标准）'
                    scale_a_dtype=scale_b_dtype='E8M0'
                    scale_a_grain=scale_b_grain='32 elements; matrix axis/layout pending'
                else:
                    aa='FP32 / token（默认）';bb='FP32 / 输出通道（默认）'
                    scale_a_dtype=scale_b_dtype='FP32'
                    scale_a_grain='token';scale_b_grain='output_channel'
                ref=link('海光接口','vendor_ops/DCU/ops/lmslim/gemm.py' if excelrow==72 else 'vendor_ops/DCU/ops/lmslim/quant_matmul.py')
                if excelrow==72:
                    implementation={'bfloat16':'torch.matmul 路径；待验收','int8':'LMSlim W8A8 接口；待依赖 / 验收','float8':'LMSlim W8A8 接口；待依赖 / 验收','float4':'海光 gemm 当前跳过；待实现 / 验收'}[element]
                else:
                    implementation='LMSlim W8A8 接口；待依赖 / 验收' if element=='int8' else '海光 quant_matmul 当前拒绝；待实现 / 验收'
                status=f'{len(M)} 组 MNK；'+('FP4 如有；' if optional else '')+implementation
                section+='| '+' | '.join([f'[{rid}](#{target})<a name="{target}"></a>',op+'<br>'+display_precision,'`'+op+'`',f'{mlabel} × {N} × {K}','—','—','1',aa,bb,ref+'<br>'+status])+' |\n'
                row={k:'' for k in projected[0]}
                row.update(requirement_id=rid,business='1 月字节送测需求',priority='P1',model=title,model_id=kind,table='byte',anchor=target,op=op,M=mlabel,N=N,K=K,batch_or_E='—',input_dtype=precision,output_dtype=outdtype,phase='—',frequency=1,legacy_compat=False,A_scale=aa,B_scale=bb,op_names=op,source_records=f"{source['workbook']}:{source['sheet']}!{excelrow}",function_reference=ref,status=status)
                new.append(row)
                for m in M:
                    expanded.append(dict(requirement_id=rid,source_row=excelrow,operator=op,M=m,N=N,K=K,sp_size=1 if excelrow==86 else '',dtype=dtype if isinstance(dtype,str) else dtype[0],w_dtype=dtype if isinstance(dtype,str) else dtype[1],compute_dtype='' if isinstance(dtype,str) else dtype[2],dst_dtype=outdtype,A_scale_dtype=scale_a_dtype,A_scale_granularity=scale_a_grain,B_scale_dtype=scale_b_dtype,B_scale_granularity=scale_b_grain,conditional_FP4=optional))
        count=number-first+1
        summaries.append(dict(model=kind,name=title,priority='P1',business='1 月字节送测需求',requirement_groups=count,first_ID=f'R{first:03}',last_ID=f'R{number:03}',expanded_cases=sum(x['source_row']==excelrow for x in expanded)))
        section+='\n'
    runtime=json.loads((AUDIT/'runtime-contract.json').read_text())
    assert runtime['repository_commit']==source['upstream']['commit']
    section+=(AUDIT/'runtime-summary.md').read_text()+'\n'
    section+='海光仓库运行入口使用 `--backend DCU`；底层 LMSlim 依赖与已有代码路径见接口链接。默认值属于本文实现约定，历史 CPU / PMD 结果按采集版本保留。\n\n'
    section+='[原始行、完整展开网格及仓库核对](debug_tuning/byte_january_20261105/README.md)\n\n'
    assert len(new)==40 and len(expanded)==1328
    assert sum(not x['conditional_FP4'] for x in expanded)==1272
    text=text.replace('# 少伯 BU GEMM 支持需求','# GEMM 支持需求')
    text=text.replace('需求来源：`shaobob0_shapes_1102/` 目录中的新版 Excel。','需求来源：`shaobob0_shapes_1102/` 新版 Excel，以及 `GPU单体测试用例 v2.1.xlsx` 的第 72–97 行（排除 MoE）。')
    text=text.replace('| 模型 |','| 模型 / 测试项 |')
    lines=''
    for x in summaries:
        lines+=f"| {x['business']} | P1 | [{x['name']}](#model-{x['model']}) | {x['requirement_groups']} | [{x['first_ID']}–{x['last_ID']}](#{x['first_ID'].lower()}) |\n"
    text=text.replace('| 后续支持 | P2 |',lines+'| 后续支持 | P2 |',1)
    text=text.replace('<a name="follow-up-p2"></a>',section+'<a name="follow-up-p2"></a>',1)
    projected=projected[:46]+new+projected[46:]
    assert [r['requirement_id'] for r in projected]==[f'R{i:03}' for i in range(1,400)]
    for m in stats['models']:
        if m['priority']=='P2':
            for field in ['first_ID','last_ID']:m[field]=re.sub(r'([Rr])(\d{3})',renumber,m[field])
    stats['models']=stats['models'][:3]+summaries+stats['models'][3:]
    stats['businesses'].insert(1,dict(name='1 月字节送测需求',priority='P1'))
    stats.pop('priority_only_P0_or_P2',None)
    stats['priorities']=['P0','P1','P2']
    stats.update(model_table_rows=399,independent_requirement_IDs=399,display_numbering='R001-R399',january_requirement_groups=40,january_expanded_cases=1328,january_required_cases=1272,january_conditional_FP4_cases=56)
    for filename,dtypes in [('basic-required.json',source['basic']['dtypes'][:3]),('basic-fp4-optional.json',['float4'])]:
        (AUDIT/filename).write_text(json.dumps(dict(cases=[dict(arg_type='default',dtype=dtypes,**({'dst_dtype':'bfloat16'} if filename=='basic-required.json' else {}),**{'K.N':source['basic']['KN'],'M':source['basic']['M']})]),ensure_ascii=False,indent=2)+'\n')
    (AUDIT/'quant-matmul.json').write_text(json.dumps(dict(quant_matmul=[dict(arg_type='llm',**{'dtype.w_dtype.compute_dtype.dst_dtype':source['quant']['dtypes'],'sp_size.hidden_size.new_hidden_size':source['quant']['sp_K_N'],'num_tokens':source['quant']['num_tokens']})]),ensure_ascii=False,indent=2)+'\n')
    with (AUDIT/'expanded-cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(expanded[0]),lineterminator='\n');w.writeheader();w.writerows(expanded)
    (AUDIT/'verification.json').write_text(json.dumps(dict(included_rows=[72,86],excluded_moe_rows=source['excluded_moe_rows'],groups=40,basic_cases=224,basic_required=168,basic_conditional_FP4=56,quant_cases=1104,implementation_policy='implementation-policy.json',basic_INT8_default_output='bfloat16',missing_scale_default='FP32 token/output_channel',MX_standard_scale='E8M0 / 32 elements',total_cases=1328,required_cases=1272,token_6192_preserved=True,excel_precision_combinations_preserved=True,CPU_reference_paths_executed=[r['operator'] for r in runtime['executions']],prepare_accepted=sum(r['accepted'] for r in runtime['prepare_records']),prepare_rejected=sum(not r['accepted'] for r in runtime['prepare_records']),GPU_tests_run=False),ensure_ascii=False,indent=2)+'\n')
    return text,projected,stats
