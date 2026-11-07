"""Portable inputs, exclusive output claims and explicit compute boundaries."""
import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)

def require_compute(allow_local=False):
    if not os.environ.get('SLURM_JOB_ID') and not allow_local:
        raise RuntimeError('Use a Slurm allocation, or explicitly allow compute on your own non-cluster workstation.')

def claim_output(path, inputs=()):
    """Never overwrite/reuse a destination or nest output into source inputs."""
    dest = Path(path).expanduser().resolve()
    if dest == Path(dest.anchor) or dest in {Path.home().resolve(), Path.cwd().resolve()}:
        raise ValueError('Choose a dedicated new output subdirectory')
    for item in inputs:
        src = Path(item).expanduser().resolve()
        if dest == src or dest.is_relative_to(src) or src.is_relative_to(dest):
            raise ValueError('Outputs must be disjoint from all input files/directories')
    # Disallow source/config/test/results/document locations in this checkout.
    checkout = PACKAGE_ROOT
    if dest.is_relative_to(checkout) and not dest.is_relative_to(checkout / 'outputs') and not dest.is_relative_to(checkout / 'data'):
        raise ValueError('Inside this checkout, use a fresh outputs/ or data/ subdirectory')
    dest.mkdir(parents=True, exist_ok=False, mode=0o700)
    return dest

def require_gpu(allow_local=False):
    require_compute(allow_local)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU fallback for full training/evaluation')

def verify_model(path):
    # The same frozen identities must be available in a wheel and a checkout.
    info = json.loads(files('truthfulness_lora').joinpath('_data/input_identities.json').read_text())
    for name, expected in info['base_files_sha256'].items():
        if digest(Path(path)/name) != expected:
            raise ValueError('Model bytes differ from the pinned research snapshot: '+name)

def bind_files(output, files):
    bindings = {str(Path(p).resolve()): digest(p) for p in files}
    dump(Path(output) / 'INPUT_BINDINGS.json', {'files': bindings})
    return bindings

def verify_bindings(output):
    info = json.loads((Path(output) / 'INPUT_BINDINGS.json').read_text())
    for path, expected in info['files'].items():
        if digest(path) != expected:
            raise ValueError('Bound run input changed: ' + path)
    return info['files']
