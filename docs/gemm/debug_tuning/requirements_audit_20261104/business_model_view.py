"""Business/model tables with one independent R ID per displayed group.

Legacy G/source mappings are retained in the audit CSV only.
"""
from collections import Counter
import re
import html

BUSINESSES = [
    ('december-bu-p0', '12 月 BU 性能测试', 'P0'),
    ('follow-up-p2', '后续支持', 'P2'),
]
MODELS = [
    ('q35-fp8', 'Qwen3.5-35B-A3B FP8', ('X09',), 'P0', 'quant-q35',
     '4k/1k、s150；TP1，单卡实例，chunk 6656。'),
    ('q35-mxfp4', 'Qwen3.5-35B-A3B MXFP4', ('X10',), 'P0', 'quant-mx4',
     '128k/1k、s150；TP1，单卡实例，chunk 6656。'),
    ('qwen3-vl-8b', 'Qwen3-VL-8B BF16', ('X14',), 'P0', None,
     '主干输入 226 token、输出 300；文本 30 token + 448×448 图像；TP1/bs512。文件保留 s150 标签。'),
    ('ds-flash', 'DeepSeek-V4-Flash', ('X01','X02','X03','X04'), 'P2', 'quant-flash',
     '64k/128k × s150/s2000；各文件部署方案分别保留。'),
    ('ds-pro', 'DeepSeek-V4-Pro', ('X05','X06'), 'P2', 'quant-pro',
     '128k/1k、s2000；P 侧 a/b 两种方案，分别为 16 卡 PP2 与 8 卡方案。'),
    ('glm51-fp8', 'GLM-5.1 FP8', ('X07','X08'), 'P2', 'quant-glm',
     '128k/64k、s2000；原表头 5.2/5.1 的命名歧义保留，以第 3 行描述 5.1 为准。'),
    ('kimi26-w4a16', 'Kimi-2.6 W4A16', ('X11',), 'P2', 'quant-kimi',
     '128k/1k、s2000；包含 EAGLE3 draft。'),
    ('hy3-fp8', 'Hunyuan-V3 FP8', ('X12',), 'P2', 'quant-hy3',
     '128k/1k、s2000；激活 FP32/token/dyn，权重 scale 为 FP32*/channel，以新版标签为准。'),
    ('q397-fp8', 'Qwen3.5-397B-A17B FP8', ('X13',), 'P2', 'quant-q397',
     '128k/1k、s2000；本模型属于后续支持 P2，不在本轮三个 Qwen P0 配置内。'),
]
TABLE_ORDER = {'3':0,'2':1,'1':2,'4':3,'5':4,'6':5,'9':6,'8':7}


def row(cells):
    return '| ' + ' | '.join(str(x) for x in cells) + ' |'


def scale_cell(value):
    # Keep starred Excel scales literal; GitHub otherwise treats *@128 as a user mention.
    return '<br>'.join('<code>'+html.escape(part)+'</code>' if re.search(r'\*@[A-Za-z0-9]', part) else part
                      for part in value.split('<br>'))


def anchor(name):
    return f'<a name="{name}"></a>'


def is_moe_source(source):
    op = source['op_name']
    return (source['op_type'] == 'moe_grouped_gemm'
            or op.startswith('moe_')
            or op in {'fused_moe_gate', 'shared_gate_up_proj', 'shared_down_proj'})


def render(cases, gaps, file_to_x, manifest, canonical, effective_a, effective_b):
    source_files = {item['source']: item['file'] for item in manifest}
    owners = {x: m[0] for m in MODELS for x in m[2]}
    assert len(owners) == sum(len(m[2]) for m in MODELS) == 14
    assert set(owners) == set(file_to_x.values())
    numbered = {c['G']:c for c in cases}
    blocks, projected, summary = {}, [], []
    display_number = 0
    sorted_gaps = sorted(gaps)
    for slug, name, sources, prio, quant_anchor, config in MODELS:
        business = BUSINESSES[0][1] if prio == 'P0' else BUSINESSES[1][1]
        entries = []
        for c in sorted(cases, key=lambda c:(TABLE_ORDER[c['table']],c['G'])):
            rs = [r for r in c['sources'] if file_to_x[r['file']] in sources
                  and not is_moe_source(r)]
            if rs:
                entries.append((c['G'],c['key'],rs,c['table'],c['legacy_compat'],
                                c['cells'][-2],c['cells'][-1]))
        for i,(key,rs) in enumerate(sorted_gaps,1):
            rs = [r for r in rs if file_to_x[r['file']] in sources
                  and not is_moe_source(r)]
            if rs:
                entries.append((f'gap-{i:03}',key,rs,'gap',False,
                                '待 grouped 接口映射','待范围确认 / 接口映射与验证'))
        block = anchor('model-'+slug)+'\n'
        block += f'\n### {name}\n\n'
        block += config+'\n\n'
        block += '需求来源：' + '、'.join('`'+source_files[x]+'`' for x in sources) + '。\n\n'
        actual_records = sum(len(rs) for _,_,rs,_,compat,_,_ in entries if not compat)
        compat_records = sum(len(rs) for _,_,rs,_,compat,_,_ in entries if compat)
        summary.append(dict(model=slug,name=name,priority=prio,business=business,
            requirement_groups=len(entries),first_ID=f'R{display_number+1:03}',last_ID=f'R{display_number+len(entries):03}',
            numbered_rows=sum(t!='gap' for _,_,_,t,_,_,_ in entries),
            grouped_gaps=sum(t=='gap' for _,_,_,t,_,_,_ in entries),
            actual_excel_sources=actual_records,compatibility_source_reuse=compat_records))
        block += row(['编号','算子 / 精度','层 / 算子名','M × N × K','batch / E','阶段','来源数','A scale','B scale','接口参考 / 状态'])+'\n'
        block += row(['---']*10)+'\n'
        for item_id,key,rs,table,compat,interface,status in entries:
            display_number += 1
            requirement_id = f'R{display_number:03}'
            target = requirement_id.lower()
            label = f'[{requirement_id}](#{target})' + anchor(target)
            ops = sorted({r['op_name'] for r in rs})
            phase = '/'.join(p for p in ['prefill','decode']
                             if any(('prefill' if r['sheet'].startswith('prefill') else 'decode')==p for r in rs))
            has_a = any(any(k.startswith('act_quant (') and v for k,v in r.items()) for r in rs)
            has_b = any(any(k.startswith('weight_quant (') and v for k,v in r.items()) for r in rs)
            aa = effective_a(rs) if has_a else '—'
            bb = effective_b(rs) if has_b else '—'
            dtype = key[5].replace('*','×')+'→'+key[6]
            if compat:
                dtype += '（兼容）'
            cellop = key[0]+'<br>'+dtype
            block += row([label,cellop,'<br>'.join('`'+x+'`' for x in ops),
                ' × '.join(map(str,key[1:4])),key[4] if key[0]!='linear' else '—',
                phase,len(rs),scale_cell(aa),scale_cell(bb),interface+'<br>'+status])+'\n'
            projected.append(dict(requirement_id=requirement_id,business=business,priority=prio,model=name,model_id=slug,
                legacy_row_id=item_id,legacy_G=item_id if table!='gap' else '',table=table,anchor=target,
                op=key[0],M=key[1],N=key[2],K=key[3],batch_or_E=key[4],
                input_dtype=key[5],output_dtype=key[6],phase=phase,frequency=len(rs),
                legacy_compat=compat,A_scale=aa,B_scale=bb,op_names=';'.join(ops),
                source_records=';'.join(f"{file_to_x[r['file']]}:{r['sheet']}!{r['row']}" for r in rs),
                function_reference=interface,status=status))
        blocks[slug]=block

    # Every retained non-MoE source stays assigned to its original group/model.
    expected = Counter()
    for c in cases:
        expected.update((c['G'],f"{file_to_x[r['file']]}:{r['sheet']}!{r['row']}")
                        for r in c['sources'] if not is_moe_source(r))
    for i,(_,rs) in enumerate(sorted_gaps,1):
        expected.update((f'gap-{i:03}',f"{file_to_x[r['file']]}:{r['sheet']}!{r['row']}")
                        for r in rs if not is_moe_source(r))
    actual = Counter((r['legacy_row_id'],source) for r in projected for source in r['source_records'].split(';'))
    assert actual == expected and all(v==1 for v in actual.values())
    included_G = {c['G'] for c in cases if any(not is_moe_source(r) for r in c['sources'])}
    assert {r['legacy_G'] for r in projected if r['legacy_G']} == included_G
    assert [r['requirement_id'] for r in projected] == [f'R{i:03}' for i in range(1,len(projected)+1)]
    assert sum(r['actual_excel_sources'] for r in summary)==1566
    assert sum(r['compatibility_source_reuse'] for r in summary)==0
    assert all(r['op'] != 'grouped_gemm' for r in projected)
    assert {r['priority'] for r in projected}=={'P0','P2'}

    text = '# 少伯 BU GEMM 支持需求\n\n需求来源：`shaobob0_shapes_1102/` 目录中的新版 Excel。\n\n'
    text += anchor('business-model-index') + '\n' + anchor('p0-快速索引') + '\n\n'
    text += '业务与模型快速索引：\n\n'
    text += row(['业务','优先级','模型','需求组数','编号范围'])+'\n'+row(['---']*5)+'\n'
    for item in summary:
        text += row([item['business'],item['priority'],f"[{item['name']}](#model-{item['model']})",
                     item['requirement_groups'],f"[{item['first_ID']}–{item['last_ID']}](#{item['first_ID'].lower()})"])+'\n'
    text += '\n'
    for biz_id,biz_name,prio in BUSINESSES:
        text += anchor(biz_id)+f'\n\n## {biz_name}（{prio}）\n\n'
        for slug,_,_,model_prio,_,_ in MODELS:
            if model_prio == prio:
                text += blocks[slug]+'\n'
    text += '\n[来源与核对记录](debug_tuning/requirements_audit_20261104/README.md)\n\n'
    used_references=set(re.findall(r'\]\[([^]]+)\]',text))
    references=[line for line in re.findall(r'^\[fn-[^]]+\]:.*$',canonical,re.M)
                if line.split(']:',1)[0][1:] in used_references]
    text+='\n'.join(references)+'\n'
    # Explicit named anchors survive this GitLab instance's HTML sanitizer.
    names=re.findall(r'<a name="([^"]+)"',text)
    assert len(names)==len(set(names))
    assert set(re.findall(r'\]\(#([^)]*)\)',text)) <= set(names)
    stats=dict(layout='business -> model',businesses=[dict(name=n,priority=p) for _,n,p in BUSINESSES],
        models=summary,model_table_rows=len(projected),global_numbered_IDs=len(numbered),
        global_grouped_gaps=len(gaps),actual_excel_source_records=1566,compatibility_source_reuse=0,
        excluded_MoE_actual_sources=960,excluded_MoE_compatibility_source_reuse=304,
        each_source_projected_to_exactly_one_model=True,shared_ID_scale_and_frequency_filtered_by_model=True,
        independent_requirement_IDs=len(projected),display_numbering=f'R001-R{len(projected):03}',
        legacy_G_mappings_in_audit_only=True,priority_only_P0_or_P2=True)
    from byte_january_view import integrate
    return integrate(text,projected,stats)
