#!/usr/bin/env python3
"""Summarize every served position against the same-length BF16/FP32 reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('served_reference', type=Path)
args = parser.parse_args()
payload = args.served_reference.read_bytes()
report = json.loads(payload)
rows = []
for record in report['records']:
    scored = record['rows']
    if not scored:
        continue
    assert len(scored) == record['length'] - 1
    native = [abs(x['native_logprob'] - x['transformers_fp32_logprob']) for x in scored]
    own = [abs(x['transformers_bf16_logprob'] - x['transformers_fp32_logprob']) for x in scored]
    assert all(math.isfinite(x) for x in native + own)
    rows.append({'prompt_ids': record['length'], 'scored_rows': len(scored),
                 'worst_served_minus_fp32': max(native),
                 'median_served_minus_fp32': statistics.median(native),
                 'worst_bf16_minus_fp32_same_length': max(own),
                 'per_length_worst_within_measured_bf16_distance': max(native) <= max(own)})
assert sum(x['scored_rows'] for x in rows) == 4511
print(json.dumps({'member_sha256': hashlib.sha256(payload).hexdigest(),
                  'median_definition': 'statistics.median over all scored positions of the length',
                  'rule': 'max(abs(served-FP32)) <= max(abs(BF16-FP32)) at the same length; no extra allowance',
                  'rows': rows, 'scored_rows': 4511,
                  'per_length_rule_all_passed': all(x['per_length_worst_within_measured_bf16_distance'] for x in rows)}, indent=2))
