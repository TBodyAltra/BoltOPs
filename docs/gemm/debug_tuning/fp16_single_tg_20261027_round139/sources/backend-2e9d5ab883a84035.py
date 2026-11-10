"""Private post-WASP prototype: stream aligned K64 reads from full BK128 LDS.
Only the proven two-consumer, two-dot, no-consumer-ebarrier shape is accepted.
Default compiler behavior is unchanged unless the explicit option enables this.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

SSA = r'%[\w.]+'


def _tensor_shapes(text):
    return [list(map(int, s.split('x'))) for s in re.findall(r'tensor<([0-9]+(?:x[0-9]+)*)xf16', text)]


def _definition(line):
    m = re.match(r'\s*(' + SSA + r'(?:,\s*' + SSA + r')*) = (.*)', line)
    return (re.findall(SSA, m[1]), m[2]) if m else ([], '')


def _consumer(part, aliases, role):
    if 'hcu.consumer_pingpong' in part:
        raise ValueError('partial K reads require the two-dot path without consumer ebarriers')
    assert '__pkl_' not in part
    lines = part.splitlines(True)
    defs = {}
    for i, line in enumerate(lines):
        values, rhs = _definition(line)
        for j, value in enumerate(values):
            assert value not in defs, value
            defs[value] = (i, j, rhs)
    dots = [i for i, line in enumerate(lines) if ' = tt.dot ' in line]
    assert len(dots) == 2, 'prototype accepts exactly two K64 dots per consumer'
    releases = [i for i, line in enumerate(lines) if ' = rocdl.hcu.s.abarrier.arrive ' in line]
    assert len(releases) == 2
    for i in releases:
        value = _definition(lines[i])[0][0]
        assert len(re.findall(re.escape(value) + r'(?![\w.])', part)) == 1
    removed = set()
    roots = {}

    def trace(value):
        assert value in defs, ('unknown operand', value)
        i, result, rhs = defs[value]
        op = rhs.split(' ', 1)[0]
        removed.add(i)
        if op == 'ttg.local_load':
            m = re.match(r'ttg.local_load (' + SSA + r')(.*?) : (!ttg.memdesc<[^>]+>) -> (tensor<.*>)( loc\([^\n]+\))$', rhs)
            assert m, rhs
            src, attrs, src_type, dst_type, loc = m.groups()
            payload = [x.strip() for x in src_type[len('!ttg.memdesc<'):-1].split(',')]
            if payload[1] in aliases:
                opidx = aliases[payload[1]]
                shape = list(map(int, payload[0].split('xf16')[0].split('x')))
                assert len(shape) == 2 and shape[1 - opidx] == 128
                roots[value] = dict(value=value, src=src, attrs=attrs, src_type=src_type,
                                    parts=payload, shape=shape, opidx=opidx, loc=loc)
                return roots[value], 0, 128
            ai, _, arhs = defs[src]
            assert arhs.startswith('ttg.local_alloc '), arhs
            removed.add(ai)
            parent = re.match(r'ttg.local_alloc (' + SSA + r')', arhs)
            assert parent
            return trace(parent[1])
        if op == 'ttg.convert_layout':
            return trace(re.match(r'ttg.convert_layout (' + SSA + r')', rhs)[1])
        assert op == 'tt.split', ('unsupported register view', rhs)
        trans_value = re.match(r'tt.split (' + SSA + r')', rhs)[1]
        ti, _, trhs = defs[trans_value]
        assert trhs.startswith('tt.trans ')
        removed.add(ti)
        reshape_value = re.match(r'tt.trans (' + SSA + r')', trhs)[1]
        ri, _, rrhs = defs[reshape_value]
        assert rrhs.startswith('tt.reshape ')
        removed.add(ri)
        parent = re.match(r'tt.reshape (' + SSA + r')', rrhs)[1]
        root, offset, length = trace(parent)
        assert result in (0, 1) and length == 128
        shapes = _tensor_shapes(rrhs)
        assert len(shapes) == 2
        nonk = root['shape'][root['opidx']]
        expected = [nonk, 2, 64] if root['opidx'] == 0 else [2, 64, nonk]
        assert shapes[-1] == expected, shapes
        order = re.search(r'order = array<i32: ([0-9, ]+)>', trhs)
        assert order and list(map(int, order[1].split(','))) == ([0, 2, 1] if root['opidx'] == 0 else [1, 2, 0])
        return root, offset + result * 64, 64

    fragments = []
    for dot_index, i in enumerate(dots):
        line = lines[i]
        m = re.search(r'tt.dot (' + SSA + r'), (' + SSA + r'),', line)
        assert m
        types = line.split(' : ', 1)[1]
        ty_a, rest = types.split(' * ', 1)
        ty_b = rest.split(' -> ', 1)[0]
        reads = []
        for opidx, (operand, target_type) in enumerate(zip(m.groups(), [ty_a, ty_b])):
            root, offset, length = trace(operand)
            assert root['opidx'] == opidx and (offset, length) == (dot_index * 64, 64)
            shape = list(root['shape']);shape[1 - opidx] = 64
            assert _tensor_shapes(target_type) == [shape]
            reads.append(dict(root=root, offset=offset, target_type=target_type, shape=shape))
        fragments.append((i, m, reads))
    assert len(roots) == 2
    # Every removed definition must be internal to the replaced operand graph.
    deleted_values = {v for i in removed for v in _definition(lines[i])[0]}
    for i, line in enumerate(lines):
        if i in removed or i in dots:
            continue
        if 'ttg.local_dealloc ' in line and any(v in deleted_values for v in re.findall(SSA, line)):
            removed.add(i)
            continue
        assert not (set(re.findall(SSA, line)) & deleted_values), ('external graph user', line)
    replacements = {}
    for dot_index, (i, match, reads) in enumerate(fragments):
        indent = re.match(r'\s*', lines[i])[0]
        inserted = []
        new_values = []
        for opidx, read in enumerate(reads):
            root = read['root'];parts = list(root['parts'])
            parent_shape = parts[-1] if re.fullmatch(r'\d+x\d+', parts[-1]) else 'x'.join(map(str, root['shape']))
            parts[0] = 'x'.join(map(str, read['shape'])) + 'xf16'
            if re.fullmatch(r'\d+x\d+', parts[-1]):parts[-1] = parent_shape
            else:parts.append(parent_shape)
            view_type = '!ttg.memdesc<' + ', '.join(parts) + '>'
            offsets = [0, 0];offsets[1 - opidx] = read['offset']
            desc = f'%__pkl_desc_{dot_index}_{opidx}'
            value = f'%__pkl_val_{dot_index}_{opidx}'
            inserted.append(f"{indent}{desc} = ttg.memdesc_subslice {root['src']}[{offsets[0]}, {offsets[1]}]{root['attrs']} : {root['src_type']} -> {view_type}{root['loc']}\n")
            inserted.append(f"{indent}{value} = ttg.local_load {desc}{root['attrs']} : {view_type} -> {read['target_type']}{root['loc']}\n")
            new_values.append(value)
        # No producer may refill either full tile until BOTH K64 reads are done.
        if dot_index == len(fragments) - 1:inserted.extend(lines[j] for j in releases)
        updated = lines[i][:match.start()] + 'tt.dot ' + new_values[0] + ', ' + new_values[1] + ',' + lines[i][match.end():]
        inserted.append(updated);replacements[i] = ''.join(inserted)
    output = ''.join(replacements.get(i, line) for i, line in enumerate(lines) if i not in removed and i not in releases)
    assert output.count(' = tt.dot ') == part.count(' = tt.dot ') == 2
    assert output.count(' = ttg.local_load ') == 4
    assert not re.search(r'ttg.local_alloc\s+%', output)
    # Phase/index/ready operations and the producer-owned descriptors are intact.
    for token in ['rocdl.hcu.s.abarrier.try.wait', 'arith.xori', 'scf.yield']:
        before = [l.strip() for l in part.splitlines() if token in l]
        after = [l.strip() for l in output.splitlines() if token in l]
        assert before == after
    return output, dict(consumer=role,root_descriptors=[x['src_type'] for x in roots.values()],
                        k_ranges=[[0, 64], [64, 128]],dot_order_preserved=True,
                        register_graph_nodes_removed=len(removed),new_local_reads=4,
                        release_after_all_fragment_reads=True,producer_full_bk=128,
                        original_parent_allocation_shapes_preserved=True)


def transform(source):
    aliases = {m[1]: int(m[2]) for m in re.finditer(r'^(#\w+) = #ttg.mls_shared<opIdx = ([01]),', source, re.M)}
    assert len(aliases) == 2
    markers = list(re.finditer(r'^    partition([012])\(', source, re.M))
    assert [m[1] for m in markers] == ['0', '1', '2']
    prefix = source[:markers[1].start()]
    parts = [source[markers[1].start():markers[2].start()], source[markers[2].start():]]
    rewritten = [];proof = []
    for role, part in enumerate(parts, 1):
        text, record = _consumer(part, aliases, role);rewritten.append(text);proof.append(record)
    result = prefix + ''.join(rewritten)
    for token in ['amdg.matrix_load_to_local', 'ttg.local_alloc : ()', 'rocdl.hcu.s.abarrier.init', 'rocdl.hcu.s.abarrier.seq', 'rocdl.hcu.s.abarrier.inv']:
        assert [l.strip() for l in source.splitlines() if token in l] == [l.strip() for l in result.splitlines() if token in l]
    assert result.count(' = tt.dot ') == source.count(' = tt.dot ') == 4
    return result, dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        transformed_sha256=hashlib.sha256(result.encode()).hexdigest(),
                        consumers=proof,global_full_bk128_transfers_unchanged=True,
                        no_new_allocation=True,logical_dot_k=64)


def normalize_mls(source):
    # Equivalent default-single-CTA spelling, also used by current BoltOps G171.
    pattern = r'#ttg.mls_shared<opIdx = (\d+), mlsTile = \[(\d+), (\d+)\], elemBitWidth = (\d+), elemBitTyKind = none, alt2Kind = (\d+), version = (\d+), order = \[(\d+), (\d+)\], CTALayout = #ttg<>>'
    def sub(match):
        a, b, c, d, e, f, g, h = match.groups()
        return f'#ttg.mls_shared_import<{a}, {b}, {c}, {d}, 0, {e}, {f}, {g}, {h}>'
    result, count = re.subn(pattern, sub, source)
    assert count == 2 and '#ttg.mls_shared<' not in result
    return result, count


def rewrite_module(module):
    from triton._C.libtriton import ir
    text, proof = transform(str(module))
    text, normalized = normalize_mls(text)
    proof["default_single_cta_import_aliases"] = normalized
    dump = os.environ.get('TRITON_PARTIAL_K_DUMP_DIR')
    with tempfile.TemporaryDirectory(prefix='pmd-partial-k-') as temp:
        path = Path(temp) / 'partial.ttgir';path.write_text(text)
        rewritten = ir.parse_mlir_module(str(path), module.context)
        rewritten.context = module.context
        assert rewritten.verify()
    if dump:
        directory = Path(dump);directory.mkdir(parents=True, exist_ok=True)
        (directory/'before.ttgir').write_text(str(module));(directory/'after.ttgir').write_text(text)
        (directory/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    return rewritten, proof
