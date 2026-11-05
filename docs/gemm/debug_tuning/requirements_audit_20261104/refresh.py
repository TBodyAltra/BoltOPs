"""Rebuild the requirements document from local XLSX, retaining historical IDs.

Usage: python refresh.py --excel-dir PATH --old-excel-dir PATH
Requires openpyxl. Does not modify XLSX, historical audits, or GPU test results.
"""
import argparse, ast, collections, csv, hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DOC = ROOT / 'docs/gemm/SHAOBO_BU_REQUIREMENTS.md'
HIST = ROOT / 'docs/gemm/debug_tuning/requirements_audit_20261023'
BASE = 'da99aa344f06a70e40bef1280928cab402ef15df'
GEMM = {'linear', 'bmm', 'moe_grouped_gemm'}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--excel-dir', type=Path, required=True)
parser.add_argument('--old-excel-dir', type=Path, required=True)
args = parser.parse_args()
import openpyxl


def csvread(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def load(folder):
    result, manifest = [], []
    for path in sorted(folder.glob('*.xlsx')):
        if path.name.startswith('~$'):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheets = []
        for sheet in wb:
            data = list(sheet.iter_rows(values_only=True))
            for idx, header in enumerate(data):
                if 'op_type' not in header or 'input_shapes' not in header:
                    continue
                sheets.append(dict(sheet=sheet.title, header_row=idx+1,
                                   context=[list(r) for r in data[:idx]]))
                for rownum, values in enumerate(data[idx+1:], idx+2):
                    r = {str(k): v for k, v in zip(header, values) if k is not None}
                    if r.get('op_type'):
                        result.append(dict(file=path.name, sheet=sheet.title, row=rownum, **r))
                break
        wb.close()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        manifest.append(dict(file=path.name, sha256=digest, sheets=sheets))
    return result, manifest


def quant(r, name):
    keys = [k for k in r if k.startswith(name + ' (')]
    return (r[keys[0]] or '').strip() if keys else ''


def norm(r):
    a, b, *_ = ast.literal_eval(r['input_shapes'])
    c = ast.literal_eval(r['output_shapes'])[0]
    assert a[-1] == b[-2] and a[-2] == c[-2] and b[-1] == c[-1], r
    dtype = tuple(x.strip() for x in r['input_dtypes'].strip('[]').split(','))
    return ('grouped_gemm' if r['op_type'] == 'moe_grouped_gemm' else r['op_type'],
            a[-2], b[-1], a[-1], a[0] if len(a) == 3 else 1,
            '*'.join(dtype), r['output_dtypes'].strip('[]'))


def pairkey(r):
    return (r['file'], r['sheet'], re.sub(r'_L\d+-L\d+', '', r['stage']),
            r['module'], r['op_name'], r['op_type'])


def cells(line):
    return [x.strip() for x in line.strip().strip('|').split('|')]


def mdrow(c):
    return '| ' + ' | '.join(c) + ' |'


old, oldmanifest = load(args.old_excel_dir)
new, manifest = load(args.excel_dir)
historical = json.loads((HIST/'model-sources.json').read_text())
oldhashes = {r['file']: r['sha256'] for r in oldmanifest}
assert all(oldhashes[r['file']] == r['sha256'] for r in historical['input_files'] if r['file'].endswith('.xlsx'))
file_to_x = {r['file']: r['source'] for r in historical['input_files'] if r['file'].endswith('.xlsx')}
file_to_x['shape_12_qwen3vl8b_226_s150.xlsx'] = 'X14'
assert set(file_to_x) == {r['file'] for r in manifest}
x_to_file = {v: k for k, v in file_to_x.items()}
P0_MODELS = [
    ('X09', 'Qwen3.5-35B-A3B FP8', 'C013', 'p0-q35-fp8'),
    ('X10', 'Qwen3.5-35B-A3B MXFP4', 'C018', 'p0-q35-mxfp4'),
    ('X14', 'Qwen3-VL-8B BF16', 'C020', 'p0-qwen3-vl-8b'),
]


def priority_models(rs):
    sources = {file_to_x[r['file']] for r in rs}
    return [name for source, name, _, _ in P0_MODELS if source in sources]


def priority(rs):
    return 'P0' if priority_models(rs) else '未指定'


for f in manifest:
    f['source'] = file_to_x[f['file']]

oldidx = {pairkey(r): r for r in old if r['op_type'] in GEMM}
newidx = {pairkey(r): r for r in new if r['op_type'] in GEMM}
assert len(oldidx) == sum(r['op_type'] in GEMM for r in old)
assert len(newidx) == sum(r['op_type'] in GEMM for r in new)
assert set(oldidx) <= set(newidx)
oldloc = {(r['file'], r['sheet'], r['row']): r for r in old}
new_gemm = [r for r in new if r['op_type'] in GEMM]
bykey = collections.defaultdict(list)
for r in new_gemm:
    bykey[norm(r)].append(r)

def loc(r):
    return f"{file_to_x[r['file']]}:{r['sheet']}!{r['row']}"


def refs(r):
    matches = re.findall(r'(X\d+):([^!;,\s]+)!(\d+)', r['source_records'])
    return [oldloc[x_to_file[x], sheet, int(row)] for x, sheet, row in matches]


base = subprocess.check_output(['git', 'show', BASE+':docs/gemm/SHAOBO_BU_REQUIREMENTS.md'], cwd=ROOT, text=True)
oldrows = {cells(l)[0]: cells(l) for l in base.splitlines() if re.match(r'^\| G\d+', l)}
legacy_cases = csvread(HIST/'case-provenance.csv')
assert set(oldrows) == {r['G'] for r in legacy_cases} and len(oldrows) == 469
maxid = max(int(g[1:]) for g in oldrows)
assert maxid == 505
cases = []
for c in legacy_cases:
    sources = [newidx[pairkey(r)] for r in refs(c)]
    assert len(sources) == int(c['frequency'])
    assert len({norm(r) for r in sources}) == 1
    k = norm(sources[0])
    oldk = norm(refs(c)[0])
    if c['table'] == '8':
        assert k[:5] == oldk[:5] and k[5:] == ('FP4E2M1*FP4E2M1', 'BF16')
    else:
        assert k == oldk, c['G']
    cases.append(dict(G=c['G'], table=c['table'], key=oldk,
                      cells=list(oldrows[c['G']]), sources=sources,
                      legacy_compat=c['table']=='8'))

# Append instead of filling the deliberately unused historical G436-G471 range.
fp4 = []
for c in [r for r in cases if r['table']=='8']:
    maxid += 1
    n = dict(c, G=f'G{maxid:03}', table='9', key=norm(c['sources'][0]),
             cells=list(c['cells']), legacy_compat=False, paired_G=c['G'])
    n['cells'][0] = n['G']
    fp4.append(n)
cases += fp4
current_keys = {c['key'] for c in cases if not c['legacy_compat']}
vl = [(k, rs) for k, rs in bykey.items() if k not in current_keys and rs[0]['file']==x_to_file['X14']]
assert len(vl) == 20 and sum(len(rs) for _, rs in vl) == 22
for k, rs in sorted(vl):
    maxid += 1
    g = f'G{maxid:03}'
    phase = '/'.join(sorted({'prefill' if r['sheet'].startswith('prefill') else 'decode' for r in rs}))
    cs = [g, ' × '.join(map(str,k[1:4])), phase, str(len(rs)), '[C020](#c020)',
          '[gemm_a16w16][fn-a16]<br>[matmul_persistent][fn-persistent]', '待验证（新增需求）']
    cases.append(dict(G=g, table='3', key=k, cells=cs, sources=rs, legacy_compat=False))
assert len(cases) == 523 and maxid == 559


def alabel(tag):
    if not tag:
        return '不适用（BF16）'
    parts = tag.split()
    assert parts[-1] in ('dyn', 'static'), tag
    fmt, grain, dtype, dynamic = parts
    group = '1×'+grain[1:] if grain.startswith('g') else '1×K' if grain=='token' else grain
    return f'{dtype.upper()}@{group}（'+('逐 token 动态' if grain=='token' else '动态' if dynamic=='dyn' else '静态')+'）'


def blabel(tag):
    fmt, grain, dtype = tag.split()
    group = grain[1:].replace('x','×') if grain.startswith('b') else '1×'+grain[1:] if grain.startswith('g') else grain
    return f'{dtype.upper()}@{group}'


def effective_a(rs):
    groups = collections.Counter(alabel(quant(r,'act_quant')) for r in rs)
    return ' / '.join(k+(f' × {n}条' if len(groups)>1 else '') for k,n in sorted(groups.items()))


def effective_b(rs):
    # Keep model/scale combinations paired rather than merge a mixed G by shape.
    groups = collections.defaultdict(list)
    for r in rs:
        groups[file_to_x[r['file']], blabel(quant(r,'weight_quant'))].append(r)
    same = {b for _,b in groups}
    if len(same)==1:
        return next(iter(same))+'（Excel）'
    return '<br>'.join(f'{x}：{b}（{len(ss)}条）' for (x,b),ss in sorted(groups.items()))


for c in cases:
    if c['table']=='2':
        c['cells'][4] = effective_a(c['sources'])
        c['cells'][5] = effective_b(c['sources'])
    elif c['table']=='6':
        c['cells'][4] = effective_b(c['sources'])+'；packing/zero-point 待确认'
    elif c['table'] in ('8','9'):
        c['cells'][5] = effective_a(c['sources'])
        c['cells'][6] = effective_b(c['sources'])
        if c['table']=='9':
            c['cells'][8] = '待 grouped 入口；dense 参考 [gemm_afp4wfp4][fn-fp4dense]'
            c['cells'][9] = '待 grouped 适配与验证'

# Updated source audit, with one row per logical support requirement.
provenance = []
for c in cases:
    k, rs = c['key'], c['sources']
    provenance.append(dict(G=c['G'], table=c['table'], priority=priority(rs),
        priority_models=';'.join(priority_models(rs)), op=k[0], M=k[1], N=k[2], K=k[3],
        batch_or_E=k[4], input_dtype=k[5], output_dtype=k[6], frequency=len(rs),
        legacy_compat=c['legacy_compat'], source_input_dtype=norm(rs[0])[5],
        source_records=';'.join(loc(r) for r in rs),
        effective_A_scale=effective_a(rs) if k[5].startswith(('FP8','FP4')) else '不适用',
        effective_B_scale=effective_b(rs) if quant(rs[0],'weight_quant') else '不适用',
        quantization_pairs=json.dumps([dict(source=loc(r),stage=r['stage'],layer_count=r['layer_count'],
          act_quant=quant(r,'act_quant'),weight_quant=quant(r,'weight_quant')) for r in rs],ensure_ascii=False),
        paired_legacy_G=c.get('paired_G','')))
csvwrite(OUT/'case-provenance.csv',provenance)

# Unnumbered grouped gaps remain a separate support-scope queue.
covered = {c['key'] for c in cases if not c['legacy_compat']}
gaps = [(k,rs) for k,rs in bykey.items() if k not in covered]
assert len(gaps)==42 and sum(len(rs) for _,rs in gaps)==80
assert all(k[0]=='grouped_gemm' for k,_ in gaps)
gaprows = [dict(op=k[0],M=k[1],N=k[2],K=k[3],batch_or_E=k[4],input_dtype=k[5],output_dtype=k[6],
               priority=priority(rs),priority_models=';'.join(priority_models(rs)),
               frequency=len(rs),A_scale=effective_a(rs),B_scale=effective_b(rs),source_records=';'.join(loc(r) for r in rs)) for k,rs in sorted(gaps)]
csvwrite(OUT/'grouped-gaps.csv',gaprows)
source_rows=[]
for r in new_gemm:
    k=norm(r)
    source_rows.append(dict(source=file_to_x[r['file']],file=r['file'],sheet=r['sheet'],row=r['row'],
      stage=r['stage'],module=r['module'],layer_count=r['layer_count'],op_name=r['op_name'],
      op_type=r['op_type'],M=k[1],N=k[2],K=k[3],batch_or_E=k[4],input_dtypes=r['input_dtypes'],
      output_dtypes=r['output_dtypes'],input_shapes=r['input_shapes'],output_shapes=r['output_shapes'],
      act_quant=quant(r,'act_quant'),weight_quant=quant(r,'weight_quant')))
csvwrite(OUT/'gemm-sources.csv',source_rows)
(OUT/'source-manifest.json').write_text(json.dumps(dict(checked_date='2026-11-04',base_commit=BASE,
    source_directory=str(args.excel_dir),input_files=manifest,
    adoption_policy='New XLSX explicit fields override historical public supplements; preserve FP8xFP4 as an explicit compatibility variant; no new GPU PASS.'),ensure_ascii=False,indent=2)+'\n')


def section(text, start, end, replacement):
    a=text.index(start);b=text.index(end,a)
    return text[:a]+replacement.rstrip()+'\n\n'+text[b:]


def count(table):
    cs=[c for c in cases if c['table']==table]
    return len(cs),sum(len(c['sources']) for c in cs)


intro='''# 少伯 BU GEMM 支持需求

原编号来源：`/xukang/GEMM支持清单.html`（2026-10-12）；本次于 2026-11-04 按新版目录 `shaobob0_shapes_1102/` 的 14 份 XLSX、39 个 sheet 更新。原 G/C 编号保留，不填补历史删除留下的编号空洞。原 HTML SHA256：`187673895b820bfb89d6b2890ffb35fa378a183e8922238a8f96df2ff738d11c`。

新版 Excel 共 2526 条 GEMM/BMM/grouped 来源：489 个已编号需求覆盖 2446 条，另有 42 个未编号 grouped shape 覆盖 80 条。按用户要求另保留表 8 的 34 个 FP8×FP4 兼容需求（304 条对应来源），因此主表共 **523 行支持需求**，表内频次合计 2750；兼容表与新版 FP4×FP4 表复用同一来源谱系，不能把 2750 当作新版 Excel 独立行数。

新增表 9 的 FP4×FP4 为 G506–G539；新增 Qwen3-VL-8B BF16 linear 为 G540–G559。所有 scale、量化粒度和动态/静态标签以新版 Excel 显式字段为准；原 FP8×FP4 需求与编号保留，其 scale 同步采用新版 E8M0/K32。

接口级 smoke 沿用历史证据；**本次未运行 GPU 测试，新增 FP4 grouped、新 BF16 shape，以及变更后的 scale 组合均未因此获得 PASS。** “接口 smoke 通过”不表示本行完整 MNK/E/scale 已验证或已调优。实验记录规范见 [调优记录](debug_tuning/TUNING.md)。逐行来源、原始量化标签与校验见 [本次核对记录](debug_tuning/requirements_audit_20261104/README.md) / [编号映射](debug_tuning/requirements_audit_20261104/case-provenance.csv)。
'''
text=section(base,'# 少伯 BU GEMM 支持需求','## 记录口径',intro)
text=text.replace('M/N/K 沿用 HTML 中的激活行数、输出通道数和规约维，不据此推断矩阵 stride。', '旧编号的 M/N/K 保留，新增行按新版 Excel 的逻辑 shape 记录激活行数、输出通道数和规约维；不据此推断矩阵 stride。')
text=text.replace('有量化的表分别记录 Excel 激活 scale 与公开 checkpoint 的权重 scale；缺失值记“未给出”，不能从 FP8/MXFP4 总称或“离线”推断 dtype。验收字段按 Excel 优先、缺失项网补采用；“网补”须标来源/参考版本，不表示 BU 实测或已通过验证。网上未查到相同对象的信息时仍保留缺失，不能跨模型或 draft 套填。',
 '激活/权重 scale、group size、动态性采用新版 Excel 的 act_quant / weight_quant；已明确字段不再标作“网补”。FP32*/BF16* 的星号原样保留，Excel 未解释其含义；不擅自改为无星号 dtype，也不据此确定物理存储/转换格式。仅缺失字段可参考历史固定 revision 的公开补充，须标注来源。')
text=text.replace('表 8 只关注单次 grouped GEMM','表 8/9 只关注单次 grouped GEMM')
text=text.replace('ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 469 行中','ResNet50 INT8 隐式 GEMM 卷积 24 类和末端 FC 不在这 523 行中')
index='''## 分类索引

| 表 | 算子 | A × B → 输出 | 需求行数 | 来源次数 |
| --- | --- | --- | ---: | ---: |
'''
labels={'1':('bmm','BF16 × BF16 → BF16'),'2':('linear','FP8E4M3 × FP8E4M3 → BF16'),'3':('linear','BF16 × BF16 → BF16'),'4':('linear','FP32 × FP32 → FP32'),'5':('linear','BF16 × BF16 → FP32'),'6':('linear','BF16 × INT4 → BF16'),'8':('grouped_gemm（兼容保留）','FP8E4M3 × FP4E2M1 → BF16'),'9':('grouped_gemm（新版）','FP4E2M1 × FP4E2M1 → BF16')}
for t,(op,dt) in labels.items():
    n,f=count(t);index+=f'| [表 {t}](#table-{t}) | {op} | {dt} | {n} | {f} |\n'
index+='\n表 8 的 304 次为兼容来源复用，表 9 的 304 次属于新版 Excel 实际来源。\n'
text=section(text,'## 分类索引','<a id="table-1"></a>',index)

# Regroup dense A scale while retaining prior anchors for incoming links.
head2='''<a id="table-2"></a>

## 表 2：linear · FP8E4M3 × FP8E4M3 → BF16

207 行需求，1206 条新版 Excel 来源，原 G 编号不变。A/B scale 均采用新版显式标签；逐行精确 act_quant/weight_quant 配对保存在本次编号映射中。

DeepSeek dense A 改为 FP32/K128/动态；B 仍为 E8M0/N128/K128。GLM B 为 FP32*/N128/K128；Qwen FP8 B 为 BF16/N128/K128。Hy3 A 为 FP32/逐 token 动态，B 为 FP32*/channel，替换历史公开候选的 BF16 scalar/static 口径；channel 的物理轴/布局及星号含义仍待确认。

同一 G 混合不同模型时，B scale 按 X 来源分别列出。标注来自 Excel 不代表 scale 物理布局/stride、量化连线或新组合已通过 GPU 测试；尤其 Hy3 的 channel scale 需单独核对接口适配。

### 按 A scale dtype 与 block size 分组

| 分组 | A scale | 需求行数 | 来源次数 |
| --- | --- | ---: | ---: |
'''
groups=collections.defaultdict(list)
for c in [c for c in cases if c['table']=='2']:
    tags={quant(r,'act_quant') for r in c['sources']}
    group='5' if len(tags)>1 else '3' if ' token ' in next(iter(tags)) else '1'
    groups[group].append(c)
assert {g:len(cs) for g,cs in groups.items()}=={'1':181,'3':25,'5':1}
for g,lab in [('1','FP32@1×128，动态'),('3','FP32@1×K，逐 token 动态'),('5','同一编号含 K128 与 token 两种来源')]:
    cs=groups[g];head2+=f'| [2.{g}](#table-2-scale-{g}) | {lab} | {len(cs)} | {sum(len(c["sources"]) for c in cs)} |\n'
head2+='\n<a id="table-2-scale-2"></a>\n<a id="table-2-scale-4"></a>\n\n旧 2.2 的 E8M0/K128 已依新版归入 2.1；旧 2.4 的 per-tensor 来源已依新版 token/dyn 归入 2.3。保留旧锚点用于历史引用。\n'
header2=['编号','M × N × K','阶段','频次','激活 scale（新版 Excel）','权重 scale（新版 Excel）','模型 / 配置','BoltOps 函数 / 实现','状态']
for g,lab in [('1','A scale FP32，block size 1×128'),('3','A scale FP32，逐 token 的整行 K'),('5','同一编号含多种 A scale')]:
    cs=groups[g];head2+=f'\n<a id="table-2-scale-{g}"></a>\n\n### 2.{g} {lab}\n\n{len(cs)} 行需求，累计 {sum(len(c["sources"]) for c in cs)} 条新版来源。\n\n'+mdrow(header2)+'\n'+mdrow(['---']*len(header2))+'\n'+'\n'.join(mdrow(c['cells']) for c in sorted(cs,key=lambda c:c['G']))+'\n'
text=section(text,'<a id="table-2"></a>','<a id="table-3"></a>',head2)
# Add new BF16 rows, preserving all old implementation/status cells.
a=text.index('<a id="table-3"></a>');b=text.index('<a id="table-4"></a>',a)
s=text[a:b].replace('133 行需求，累计 326 条实例。','153 行需求，累计 348 条来源（新增 Qwen3-VL-8B 20 行 / 22 条来源）。')
s=s.rstrip()+'\n'+'\n'.join(mdrow(c['cells']) for c in cases if c['table']=='3' and int(c['G'][1:])>=540)+'\n\n'
text=text[:a]+s+text[b:]
head6='''<a id="table-6"></a>

## 表 6：linear · BF16 × INT4 → BF16

4 行需求、4 条来源，G432–G435 仍为 Kimi EAGLE3 draft。新版现已明确 weight_quant=`int4 g32 bf16*`，因此采用 K32 / BF16* scale；A 为 BF16，无 A scale。星号含义、zero-point、对称性与 packing 仍未给出，不由 Kimi 主模型或 AWQ 的格式反推。AWQ 链接只保留历史接口参考，不证明 draft 的实际格式相容。

'''+mdrow(['编号','M × N × K','阶段','频次','权重 scale（新版 Excel）','模型 / 配置','BoltOps 函数 / 实现','状态'])+'\n'+mdrow(['---']*8)+'\n'+'\n'.join(mdrow(c['cells']) for c in cases if c['table']=='6')+'\n'
text=section(text,'<a id="table-6"></a>','<a id="table-8"></a>',head6)
header8=['编号','M × N × K','分组数 E','阶段','频次','激活 scale（新版 Excel）','权重 scale（新版 Excel）','模型 / 配置','BoltOps 函数 / 实现','状态']
head8='''<a id="table-8"></a>

## 表 8：grouped_gemm · FP8E4M3 × FP4E2M1 → BF16（兼容保留）

按用户要求保留原 34 行、G472–G505、MNK/E、阶段及频次。新版 Excel 的对应 304 条来源实际已改为 FP4×FP4；本表是额外保留的 FP8×FP4 支持需求，不表示新版仍要求 A=FP8。

scale 按新版更新：A/B 均为 E8M0/K32，激活动态；DeepSeek 原 A 的 E8M0/FP32 K128、Q35 down 的 FP32/K32 均不再作为本次 scale 要求。新版 act_quant 的 fp4 数据 dtype 只用于表 9；本表有意保留 A=FP8，仅采用对应新版的 scale 粒度/dtype/动态性。

历史 grouped_gemm_a8wfp4 接口 smoke 与原 34/34 编译证据保留，但不验证本次更新的 scale 组合，完整 MNK/E/scale 的正确性和性能仍待逐项验证。来源次数 304 与表 9 共用谱系，不计作新增 Excel 行。

'''+mdrow(header8)+'\n'+mdrow(['---']*10)+'\n'+'\n'.join(mdrow(c['cells']) for c in cases if c['table']=='8')+'''

<a id="table-9"></a>

## 表 9：grouped_gemm · FP4E2M1 × FP4E2M1 → BF16（新版新增）

34 行、304 条新版 Excel 来源，编号 G506–G539。各行与表 8 按相同 MNK/E/来源顺序对应，具体成对 G 编号见本次编号映射的 paired_legacy_G。A/B scale 均为 E8M0/K32，A 为动态量化。

当前仓库的 gemm_afp4wfp4 是 dense FP4×FP4 参考，不能据此宣称已有等长/ragged grouped 入口或本需求 PASS。所有新增行均待 grouped 接口适配与验证。FP4 packing/nibble 顺序、scale 实际布局、offsets/空组规则须由调用数据明确；此处只记录逻辑需求。

'''+mdrow(header8)+'\n'+mdrow(['---']*10)+'\n'+'\n'.join(mdrow(c['cells']) for c in cases if c['table']=='9')+'\n'
text=section(text,'<a id="table-8"></a>','<a id="requirements-cross-check"></a>',head8)

cross='''<a id="requirements-cross-check"></a>

## 新版 Excel 与历史公开补充核对（2026-11-04）

新版 act_quant / weight_quant 显式标签是本次需求依据。下表的公开信息沿用 2026-10-23 审计中固定 revision 的快照，仅说明历史补充与新版的关系；本次没有重新检索公开 main，也不将候选 checkpoint 等同于 BU 实际部署。历史证据见 [2026-10-23 核对记录](debug_tuning/requirements_audit_20261023/REPORT.md)。

| 模型 / 历史来源 | 新版激活量化 / scale | 新版权重量化 / scale | 与历史公开补充的关系 |
| --- | --- | --- | --- |
'''
public_specs=[
 ('quant-flash','DeepSeek-V4-Flash','DeepSeek-V4-Flash','dense：FP8、FP32/K128/dyn；grouped：FP4、E8M0/K32/dyn','dense：FP8、E8M0/N128/K128；grouped：FP4、E8M0/K32','B 与旧公开快照相符；dense A E8M0→FP32、grouped A FP8→FP4 是 Excel 修订。'),
 ('quant-pro','DeepSeek-V4-Pro','DeepSeek-V4-Pro','同 Flash','同 Flash','新版 grouped FP4×FP4 列表 9；FP8×FP4 作为兼容需求保留表 8。'),
 ('quant-glm','GLM-5.1-FP8','GLM-5.1-FP8','FP8、FP32/K128/dyn，含 indexer_wk','FP8、FP32*/N128/K128','K/N 粒度与历史补充相符；星号仍未释义，不能当作已确认的无星号存储 dtype。'),
 ('quant-q35','Qwen3.5-35B-A3B-FP8','Qwen3.5-35B-A3B-FP8','FP8、FP32/K128/dyn，含 o_proj','FP8、BF16/N128/K128','新版已明确这些字段，原“网补”改为 Excel 来源；checkpoint 与框架转换后的 scale dtype 仍需分开。'),
 ('quant-q397','Qwen3.5-397B-A17B-FP8','Qwen3.5-397B-A17B-FP8','同 Q35 FP8','FP8、BF16/N128/K128','同一 G 混合 DS/Qwen 时，B dtype 按 X 来源保留两种组合。'),
 ('quant-hy3','Hy3-FP8 / Hunyuan-V3','Hy3-FP8','FP8、FP32/token/dyn，含 o_proj','FP8、FP32*/channel','替换旧候选的 BF16 scalar 权重 / FP32 scalar 静态激活；粒度与动态性变化不能只用 scale cast 解释。'),
 ('quant-mx4','Qwen3.5-35B-A3B-MXFP4','Qwen3.5-35B-A3B-MXFP4','grouped：FP4、E8M0/K32/dyn','grouped：FP4、E8M0/K32','新版 A FP4 与历史 AMD 候选方向一致；表 8 的 A FP8 是额外兼容需求。'),
 ('quant-kimi','Kimi-K2.6 / W4A16','Kimi-K2.6','BF16，无 A scale','INT4、BF16*/K32，含 EAGLE3 draft','新版补齐 draft 的 g32/BF16*；对称性、zero-point 与 packing 未由标签确定，不直接认定为 AWQ。'),
]
for anchor,label,needle,aa,bb,note in public_specs:
    url=next(m['config_url'] for m in historical['models'] if needle in m['model'])
    cross+=mdrow([f'<a id="{anchor}"></a>[{label}]({url})',aa,bb,note])+'\n'
cross+='''
同名 quantize_o_proj 与 o_proj 的 shape 衔接在 Hy3/Qwen 中仍有 18 处需说明；Excel 未给 tensor ID，不能假定相邻同名记录必定直连，也不能据此擅改 MNK。按 GEMM 自身 act_quant 标签记录需求，同时保留连线问题。grouped 量化输出与各专家 token 总数之间的复制/padding/路由映射也未给齐，不能只凭总元素关系推断 ragged offsets。

FP32*/BF16* 共涉及 248 条 GEMM 来源，原工作簿没有星号解释。FP4/INT4 packing、nibble 顺序、scale 物理布局/stride、分组 offsets/空组规则，以及部署 checkpoint/框架 revision，均不由量化标签直接确定。逻辑需求已采用新标签，物理 ABI 与实际数据链路仍须单独确认。

G171 的当前定位、stage/layer_count 和原始量化标签见新编号映射；旧表内的行号和候选量化输出位置不应当作新版坐标。Pro a 的 PP 标签与 layer_count 修订已原样保存在 gemm-sources.csv，不按 layer_count 重算频次。

### 原始文件与部署方案

X01–X13 沿用原文件编号，新增 Qwen3-VL 为 X14；SHA256、完整 sheet 描述见 [新版来源清单](debug_tuning/requirements_audit_20261104/source-manifest.json)。

| 来源 | 新版文件 | prefill / P 侧方案（原文摘要） |
| --- | --- | --- |
'''
for f in sorted(manifest,key=lambda f:f['source']):
    sheet=next(s for s in f['sheets'] if s['sheet'].startswith('prefill'))
    deployment=str(sheet['context'][3][0]).replace('|','/').split(' → ')[0]
    cross+=mdrow([f['source'],'`'+f['file']+'`',deployment])+'\n'
cross+='''
### 尚未纳入 G 编号表的 grouped GEMM 需求

保留原 42 个缺口（36 个 FP8×FP8、70 条来源；6 个 BF16×INT4、10 条来源），不擅自升级接口状态。scale/量化标签同步更新为新版 Excel，尤其 Hy3 的 token/channel 与 Kimi 的 BF16*。以下全部待范围确认 / grouped 接口映射与验证；量化值、scale 与分组索引由算子测试预先准备，不扩展为路由/激活/fused_experts 包装验收。

逐条原文件/sheet/行号见 [新版缺口映射](debug_tuning/requirements_audit_20261104/grouped-gaps.csv)。

'''+mdrow(['M × N × K','E','A × B → 输出','A scale（新版）','B scale（新版）','X 来源','记录数'])+'\n'+mdrow(['---']*7)+'\n'
for k,rs in sorted(gaps):
    cross+=mdrow([' × '.join(map(str,k[1:4])),str(k[4]),k[5].replace('*',' × ')+' → '+k[6],effective_a(rs),effective_b(rs),'/'.join(sorted({file_to_x[r['file']] for r in rs})),str(len(rs))])+'\n'
text=section(text,'<a id="requirements-cross-check"></a>','<a id="model-config-index"></a>',cross)
# Existing C names describe model/precision combinations, not one immutable dtype for every op.
text=text.replace('原始模型名称、精度和方括号内的配置原样保留。','原始模型名称、精度和方括号内的配置原样保留。C004 的 Pro a/b 与修订后的 PP stage 以新版来源清单为准；“FP8”等模型标签不覆盖行内明确的混合 dtype。')
new_c='''### C020

Qwen3-VL-8B/BF16[主干输入226/输出300/s150；文本30+448×448图像；TP1/bs512]。沿用文件第3行的 s150 标签，需求描述实际输出长度为300；完整原文见 X14 来源清单。

'''
text=text.replace('<!-- BoltOps function definitions:',new_c+'<!-- BoltOps function definitions:')
# Refresh line anchors against actual definitions, without claiming compatibility/PASS.
fnpaths={'fn-bmm':('batched_gemm_bf16.py','batched_gemm_bf16'),
 'fn-a16':('gemm_a16w16.py','gemm_a16w16'),'fn-persistent':('matmul_persistent.py','matmul_persistent'),
 'fn-fp8-aiter':('gemm_a8w8_blockscale.py','gemm_a8w8_blockscale'),
 'fn-fp8-vllm':('w8a8_triton_block_scaled_mm.py','w8a8_triton_block_scaled_mm'),
 'fn-awq':('awq_gemm_triton.py','awq_gemm_triton'),'fn-fp8fp4':('gemm_a8wfp4.py','grouped_gemm_a8wfp4'),
 'fn-fp4dense':('gemm_afp4wfp4.py','gemm_afp4wfp4')}
for ref,(file,fn) in fnpaths.items():
    p=ROOT/'boltops/gemm/triton'/file
    num=next(i for i,l in enumerate(p.read_text().splitlines(),1) if re.match(r'^def '+re.escape(fn)+r'\(',l))
    line=f'[{ref}]: ../../boltops/gemm/triton/{file}#L{num}'
    if re.search(r'^\['+ref+r'\]:.*$',text,re.M):
        text=re.sub(r'^\['+ref+r'\]:.*$',line,text,flags=re.M)
    else:text=text.rstrip()+'\n'+line+'\n'

# Independent structural/source checks before publishing the update.
newrows={cells(l)[0]:cells(l) for l in text.splitlines() if re.match(r'^\| G\d+',l)}
assert len(newrows)==523==len(re.findall(r'^\| G\d+',text,re.M))
for c in cases:
    assert newrows[c['G']]==c['cells']
for c in legacy_cases:
    before=oldrows[c['G']];after=newrows[c['G']]
    allowed={'2':{4,5},'6':{4},'8':{5,6}}.get(c['table'],set())
    assert len(before)==len(after)
    assert all(x==y for i,(x,y) in enumerate(zip(before,after)) if i not in allowed),c['G']
all_locations={loc(r) for r in new_gemm}
current_locations=[loc(r) for c in cases if not c['legacy_compat'] for r in c['sources']]
current_locations += [loc(r) for _,rs in gaps for r in rs]
assert len(current_locations)==len(set(current_locations))==2526
assert set(current_locations)==all_locations
assert len(bykey)==531 and len(covered)==489
assert sum(len(c['sources']) for c in cases if not c['legacy_compat'])==2446
assert sum(len(c['sources']) for c in cases)==2750
for r in new_gemm:
    a,b=norm(r)[5].split('*');wt=quant(r,'weight_quant');at=quant(r,'act_quant')
    if wt:assert {'fp8':'FP8E4M3','fp4':'FP4E2M1','int4':'INT4'}[wt.split()[0]]==b
    if at:assert {'fp8':'FP8E4M3','fp4':'FP4E2M1'}[at.split()[0]]==a
assert sum('*' in quant(r,'weight_quant') for r in new_gemm)==248
for c in [c for c in cases if c['table'] in ('8','9')]:
    assert {quant(r,'act_quant') for r in c['sources']}=={'fp4 g32 e8m0 dyn'}
    assert {quant(r,'weight_quant') for r in c['sources']}=={'fp4 g32 e8m0'}

# Preserve the exact stage/layer_count changes, without multiplying row frequency.
stage_changes=sum(oldidx[k]['stage']!=newidx[k]['stage'] for k in oldidx)
layer_changes=sum(oldidx[k]['layer_count']!=newidx[k]['layer_count'] for k in oldidx)
assert stage_changes==308 and layer_changes==112

# Locate same-named quantizer/dense links, without assuming these are true dataflow edges.
quantizers={(r['file'],r['sheet'],r['stage'],r['module'],r['op_name']):r for r in new if r['op_name'].startswith('quantize_')}
issues=[]
for r in new_gemm:
    if r['op_type']!='linear':continue
    q=quantizers.get((r['file'],r['sheet'],r['stage'],r['module'],'quantize_'+r['op_name']))
    if not q:continue
    qs=ast.literal_eval(q['output_shapes'])[0];ins=ast.literal_eval(r['input_shapes'])[0]
    if qs!=ins:
        issues.append(dict(file=r['file'],sheet=r['sheet'],quantizer_row=q['row'],GEMM_row=r['row'],op_name=r['op_name'],quantizer_output_shape=str(qs),GEMM_input_shape=str(ins),note='Named-link candidate only; tensor IDs/mapping not given'))
assert len(issues)==18,len(issues)
csvwrite(OUT/'named-link-issues.csv',issues)

# Priority is a support-planning annotation; the original requirement cells above
# stay unchanged. Source membership also covers shared-model and compatibility rows.
priorities = {c['G']: priority(c['sources']) for c in cases}
sorted_gaps = sorted(gaps)
p0_cases = [c for c in cases if priorities[c['G']] == 'P0']
p0_gaps = [(i, k, rs) for i, (k, rs) in enumerate(sorted_gaps, 1) if priority(rs) == 'P0']
assert len(p0_cases) == 74 and len(p0_gaps) == 6
assert sum(c['table'] == '3' for c in p0_cases) == 47
rendered, mode, gap_number = [], None, 0
for line in text.splitlines():
    if not line.startswith('| '):
        mode = None
        rendered.append(line)
        continue
    row = cells(line)
    if row[0] == '编号':
        mode = 'numbered'
        row.insert(1, '优先级')
    elif row[:2] == ['M × N × K', 'E']:
        mode = 'gaps'
        row.insert(1, '优先级')
    elif mode and all(re.fullmatch(r':?-+:?', cell) for cell in row):
        row.insert(1, '---')
    elif mode == 'numbered':
        g = row[0]
        row[0] = f'{g}<a id="{g.lower()}"></a>'
        row.insert(1, priorities[g])
    elif mode == 'gaps':
        _, rs = sorted_gaps[gap_number]
        gap_number += 1
        row[0] += f'<a id="gap-{gap_number:03}"></a>'
        row.insert(1, priority(rs))
    rendered.append(mdrow(row))
assert gap_number == len(gaps)
text = '\n'.join(rendered) + '\n'

quick = '''## P0 快速索引

当前 P0：Qwen3.5-35B-A3B 的 FP8、MXFP4 两版，以及 Qwen3-VL-8B BF16。优先级按模型来源覆盖其全部需求，含保留的 FP8×FP4 兼容项；其他模型暂记“未指定”。点击数量可跳到对应首条，展开下方模型编号清单可逐项跳转。

| 模型 / 配置 | 优先级 | BF16 dense | FP8 dense | FP4×FP4 grouped | FP8×FP4 兼容 | 未编号 grouped |
| --- | --- | --- | --- | --- | --- | --- |
'''
details = []
for source, name, config_id, anchor in P0_MODELS:
    selected = [c for c in cases if any(file_to_x[r['file']] == source for r in c['sources'])]
    links, items = [], []
    for table, label in [('3', 'BF16 dense'), ('2', 'FP8 dense'),
                         ('9', 'FP4×FP4 grouped'), ('8', 'FP8×FP4 grouped（兼容）')]:
        cs = sorted((c for c in selected if c['table'] == table), key=lambda c: c['G'])
        links.append(f'[{len(cs)} 项](#{cs[0]["G"].lower()})' if cs else '—')
        if cs:
            items.append('- ' + label + '：' + '、'.join(f'[{c["G"]}](#{c["G"].lower()})' for c in cs))
    gs = [(i, k, rs) for i, (k, rs) in enumerate(sorted_gaps, 1)
          if any(file_to_x[r['file']] == source for r in rs)]
    links.append(f'[{len(gs)} 项](#gap-{gs[0][0]:03})' if gs else '—')
    if gs:
        items.append('- 未编号 grouped（待范围 / 接口确认）：' + '、'.join(
            f'[{k[1]}×{k[2]}×{k[3]} / E{k[4]}](#gap-{i:03})' for i, k, _ in gs))
    quick += mdrow([f'[{name}](#{config_id.lower()})', 'P0', *links]) + '\n'
    details.append(f'<a id="{anchor}"></a>\n<details>\n<summary>{name}：P0 需求编号</summary>\n\n'
                   + '\n\n'.join(items) + '\n\n</details>\n')
quick += '''
合计 **74 条已编号 P0 需求**（其中 47 条 BF16 dense、15 条 FP8 dense、6 条新版 FP4×FP4、6 条 FP8×FP4 兼容），另有 **6 个 P0 grouped 缺口**。G342/G347 为两种 Qwen35 配置共享，主表各只列一次；快速索引的逐模型数量不直接相加。

''' + '\n'.join(details)
text = text.replace('## 记录口径', quick.rstrip() + '\n\n## 记录口径', 1)
for config_id in ['C013', 'C018', 'C019', 'C020']:
    text = text.replace(f'### {config_id}\n\n', f'### {config_id}\n\n优先级：P0。\n\n')

# Confirm that priority/row anchors introduced no changes to any requirement cell.
final_rows = {}
for line in text.splitlines():
    if not re.match(r'^\| G\d+', line):
        continue
    row = cells(line)
    g = re.match(r'G\d+', row[0]).group()
    assert row[1] == priorities[g]
    final_rows[g] = [g, *row[2:]]
assert final_rows == newrows
# GitLab strips arbitrary HTML `id` attributes. Its Markdown renderer retains
# named anchors; use those for both table rows and section/model destinations.
text = re.sub(r'<a id="([^"]+)"></a>', r'<a name="\1"></a>', text)
text = re.sub(r'(?m)^### (C\d+)$',
              lambda m: f'<a name="{m[1].lower()}"></a>\n\n### {m[1]}', text)
text = text.replace('## P0 快速索引', '<a name="p0-快速索引"></a>\n\n## P0 快速索引', 1)
DOC.write_text(text)
anchors=set(re.findall(r'<a name="([^"]+)"',text))
assert not (set(re.findall(r'\]\(#([^)]*)\)',text))-anchors)
assert '<a id=' not in text
verification=dict(checked_date='2026-11-04',base_commit=BASE,xlsx_files=len(manifest),sheets=sum(len(x['sheets']) for x in manifest),
    new_GEMM_sources=len(new_gemm),new_unique_GEMM_keys=len(bykey),numbered_support_rows=len(cases),
    numbered_current_excel_rows=len(covered),numbered_current_excel_sources=2446,
    preserved_legacy_FP8xFP4_rows=34,preserved_legacy_FP8xFP4_sources=304,
    added_FP4xFP4_rows=34,added_Qwen3VL_BF16_rows=20,added_Qwen3VL_BF16_sources=22,
    numbered_frequency_including_compat=2750,unnumbered_grouped_rows=len(gaps),unnumbered_grouped_sources=80,
    legacy_ID_shape_E_phase_frequency_model_implementation_status_preserved=True,
    new_sources_covered_exactly_once_excluding_compat=True,all_quant_labels_match_source_dtypes=True,
    grouped_A_and_B_scale_E8M0_K32_from_new_excel=True,starred_weight_scale_sources=248,
    corrected_stage_sources=stage_changes,corrected_layer_count_sources=layer_changes,
    named_quantizer_dense_shape_link_issues=len(issues),A_scale_groups={g:dict(rows=len(cs),sources=sum(len(c['sources']) for c in cs)) for g,cs in groups.items()},
    numbered_tables={t:dict(rows=count(t)[0],sources=count(t)[1]) for t in labels},
    priority_policy=dict(P0_models=[name for _,name,_,_ in P0_MODELS],
                         source_files=[x_to_file[x] for x,_,_,_ in P0_MODELS],
                         other_models='未指定'),
    numbered_P0_rows=len(p0_cases),numbered_unspecified_priority_rows=len(cases)-len(p0_cases),
    numbered_P0_BF16_dense_rows=47,unnumbered_P0_grouped_rows=len(p0_gaps),
    priority_annotations_preserve_requirement_cells=True,
    custom_markdown_anchor_attribute='name',GPU_tests_run=False)
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
(OUT/'README.md').write_text('''# 新版少伯 Excel 需求更新核对

日期：2026-11-04。原文档基线 da99aa344f06a70e40bef1280928cab402ef15df。

## 更新口径

- 用户指定 P0：Qwen3.5-35B-A3B FP8、Qwen3.5-35B-A3B MXFP4、Qwen3-VL-8B BF16，按 X09/X10/X14 来源标记全部关联需求；其他模型未指定。主表 74 条 P0，未编号 grouped 另有 6 条；共享 G342/G347 不重复计数。顶部快速索引包含各类首条入口与可展开的完整编号跳转，表格新增优先级列。
- 2026-11-05 修正 GitLab 锚点兼容性：自定义锚点使用 `<a name="...">`，而不是会被该实例清洗掉的 HTML `id` 属性。G 编号、缺口条目、表格、模型配置和 P0 索引均有明确的 named target；实际 GitLab 渲染核验见 rendered-anchors-20261105.json。
- 新版 `shaobob0_shapes_1102` 的 14 份 XLSX、39 个 sheet、2526 条 GEMM/BMM/grouped 来源全部读取；忽略 Excel 的 `~$` 临时锁文件，不修改原工作簿。
- 原 469 个 G 编号不重排，原 MNK/E、阶段、频次、模型引用、接口引用与历史状态保留。G436–G471 的历史空洞不复用。
- 原 FP8×FP4 的 G472–G505 作为兼容需求保留；A/B scale 按新版统一 E8M0/K32，激活动态。新版对应来源的 A dtype 是 FP4，兼容项的 FP8 不冒充新版数据。
- 新增 FP4×FP4 grouped G506–G539；新增 Qwen3-VL BF16 linear G540–G559，C020/X14。dense FP4 参考接口不等于已有 grouped 适配，新增需求全部待验证。
- 207 个 FP8 dense 的 A scale 分为 FP32/K128 181 行、FP32/token 25 行、混合 1 行；B scale 采用新标签，混合模型按 X 来源配对。DeepSeek dense A 不再使用旧 E8M0；Hy3 不再采用旧 scalar/static 补充值。
- Kimi EAGLE3 与 grouped 的 g32/BF16* 依新标签记录；FP32*/BF16* 星号保留，packing/zero-point 等不擅自补写。旧公开采样快照仅作历史说明。
- 489 个新版已编号需求覆盖 2446 条来源，42 个未编号 grouped 缺口覆盖 80 条，合计恰为 2526；额外保留的 34 个 FP8×FP4 兼容项复用 304 条谱系，不作为新版新增来源计数。

## 尚需明确

18 处同名 quantize_o_proj 与 o_proj 的 shape 候选连线仍不一致，见 named-link-issues.csv；无 tensor ID，不能直接认定实际程序连线或擅改 MNK。grouped 仍缺路由/复制/padding/offsets 与 scale 的对应说明。
248 条来源带 FP32*/BF16*，Excel 未解释星号。scale 物理布局、FP4/INT4 packing/nibble 顺序、INT4 zero-point/对称性，以及部署 checkpoint/框架 revision 均未由本表完全确定。
本次没有运行 GPU 测试或新增任何 smoke/完整尺寸/性能 PASS；历史状态仅按原接口证据保留。

## 可追溯文件与复现

- source-manifest.json：14 份新版 XLSX 的 SHA256、X 编号、sheet 描述与部署原文。
- gemm-sources.csv：2526 条实际来源，含新版行号、stage、layer_count、原始 shape/dtype 与量化标签。
- case-provenance.csv：523 条支持需求，含兼容标志、来源 dtype、成对 G 编号、逐来源 act_quant/weight_quant；可区分支持 dtype 与新版源 dtype。
- grouped-gaps.csv：42 个未编号 grouped shape，采用新版量化字段。
- named-link-issues.csv：18 个 dense 命名连线候选问题。
- verification.json：计数、旧编号不变性、当前来源恰好一次覆盖、dtype/量化标签、scale 分组与锚点检查。
- refresh.py：使用 openpyxl 从本地两版 XLSX 重新生成。基线原文从上述 Git commit 读取，旧版 XLSX 的 SHA256 与 2026-10-23 历史审计逐个校验。

```bash
python docs/gemm/debug_tuning/requirements_audit_20261104/refresh.py \\
  --excel-dir /mnt/c/Users/xukang/Downloads/shaobob0_shapes_1102 \\
  --old-excel-dir /mnt/c/Users/xukang/Downloads/shaobob0_shapes
```

未把原始 XLSX、模型权重或 GPU 产物加入 Git。2026-10-23 的历史审计文件保留不变。
''')
# Validate relative links after all generated artifacts exist.
for target in re.findall(r'\]\(([^)]+)\)',text):
    if target.startswith(('#','http://','https://')):continue
    assert (DOC.parent/target.split('#')[0]).is_file(),target
print(json.dumps(verification,ensure_ascii=False,indent=2))
