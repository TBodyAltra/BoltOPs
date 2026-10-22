"""Reassociate G171 scale products in a pinned TTGIR snapshot.

This changes FP32 multiplication order. Numerical equivalence must be measured;
the recorded seed20 result is not a proof for exceptional values or underflow.
"""
import re


def transform(text):
    lines = text.splitlines(True)
    changed = 0
    for i, line in enumerate(lines):
        first = re.match(r'(\s*)(%accumulator_\d+) = arith.mulf %partial, (%accumulator_\d+)(.*)', line)
        if first is None:
            continue
        second = re.match(
            r'\s*(%accumulator_\d+) = arith.mulf ' + re.escape(first[2]) + r', (%accumulator_\d+)(.*)',
            lines[i + 3],
        )
        assert second is not None, 'Unexpected scale multiplication structure'
        lines[i] = ''
        lines[i + 3] = (
            first[1] + first[2] + ' = arith.mulf ' + first[3] + ', ' + second[2] + first[4] + '\n'
            + first[1] + second[1] + ' = arith.mulf %partial, ' + first[2] + second[3] + '\n'
        )
        changed += 1
    assert changed == 2, 'Expected exactly two consumer partitions'
    return ''.join(lines)
