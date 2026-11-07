"""Matched MC1/MC2 aggregation; no prompt or answer records are bundled."""
import json
import math
from pathlib import Path
import numpy as np

ARMS = ('Base', 'SFT', 'F2-Lite', 'DPO')

def metric(scores, target, kind):
    values = np.array([scores[x] for x in target['choices']], dtype=float)
    labels = np.array(target['labels'])
    if len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError('Invalid score/label arrays')
    if not set(labels) <= {0, 1} or not 0 < labels.sum() < len(labels):
        raise ValueError('Expected both true and false choices')
    if kind == 'mc1':
        tied = np.isclose(values, values.max(), atol=1e-8, rtol=0)
        return float(labels[tied].sum() / tied.sum())
    if kind != 'mc2':
        raise ValueError('Metric must be mc1 or mc2')
    masses = np.exp(values - values.max())
    return float(masses[labels == 1].sum() / masses.sum())

def summarize(records, resamples=10000, seed=42):
    """Same shared RNG draws/order as original; all six pairs are returned."""
    n = len(records['Base'])
    if n == 0 or resamples <= 0:
        raise ValueError('Nonempty records and positive resample count required')
    keys = [(r['benchmark_index'], r['question']) for r in records['Base']]
    if len({k[0] for k in keys}) != n:
        raise ValueError('Duplicate benchmark index')
    arrays = {}
    for arm in ARMS:
        rr = records[arm]
        if [(r['benchmark_index'], r['question']) for r in rr] != keys:
            raise ValueError('Paired item alignment differs')
        arrays[arm] = {}
        for m in ['mc1', 'mc2']:
            x = np.array([r[m] for r in rr], dtype=float)
            if not np.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
                raise ValueError('Invalid metric value')
            arrays[arm][m] = x
    indices = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    arms, paired = {}, {}
    for arm in ARMS:
        arms[arm] = {m: {'mean': float(x.mean()), 'bootstrap95': np.quantile(x[indices].mean(1), [.025, .975]).tolist()} for m, x in arrays[arm].items()}
    for i, earlier in enumerate(ARMS):
        for later in ARMS[i+1:]:
            key = later + '-minus-' + earlier
            paired[key] = {}
            for m in ['mc1', 'mc2']:
                x = arrays[later][m] - arrays[earlier][m]
                paired[key][m] = {'mean': float(x.mean()), 'paired_bootstrap95': np.quantile(x[indices].mean(1), [.025, .975]).tolist()}
    return {'questions': n, 'arms': arms, 'paired_differences': paired,
            'bootstrap': {'seed': seed, 'resamples': resamples, 'unit': 'paired question', 'ci': 'nominal percentile 2.5/97.5; no seed or multiplicity adjustment'}}

def read_records(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
