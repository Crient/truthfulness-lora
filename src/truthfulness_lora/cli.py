"""Subcommands defer heavy imports until after argument validation."""
import argparse
import os
from pathlib import Path
from .runtime import claim_output, bind_files, dump, require_compute

def parser():
    p = argparse.ArgumentParser(description='Private research release candidate; no default training action')
    sub = p.add_subparsers(dest='command', required=True)
    def command(name, help_text):
        q=sub.add_parser(name,help=help_text)
        q.add_argument('--output', required=True, help='New, nonexistent output directory')
        q.add_argument('--allow-local-compute',action='store_true',help='Only for a dedicated non-cluster workstation; cluster users use Slurm')
        return q
    command('stage-inputs','Explicitly download pinned inputs after license review')
    for name in ['prepare-halueval','freeze-benchmark']:
        q=command(name,'Prepare pinned local inputs');q.add_argument('--source',required=True)
    q=command('prepare-dpo','Rebuild the exploratory preference selection on CPU')
    q.add_argument('--model',required=True);q.add_argument('--source-parquet',required=True);q.add_argument('--benchmark',required=True)
    q.add_argument('--acknowledge-exploratory-limitations',action='store_true',required=True)
    q=command('train-supervised','SFT or F2-Lite using extracted original notebook helpers')
    q.add_argument('--recipe',choices=['sft','f2-lite'],required=True);q.add_argument('--model',required=True);q.add_argument('--split',required=True)
    q=command('train-dpo','SFT-initialized DPO with original atomic optimizer-boundary checkpoints')
    q.add_argument('--model',required=True);q.add_argument('--sft',required=True);q.add_argument('--prepared',required=True)
    q.add_argument('--acknowledge-exploratory-limitations',action='store_true',required=True)
    q=command('evaluate','Matched four-arm native-chat likelihood evaluation')
    for n in ['model','sft','f2','dpo-run','prepared','benchmark']:q.add_argument('--'+n,required=True)
    q=command('summarize','Aggregate locally available private per-question records')
    q.add_argument('--records',required=True)
    return p

def environment(args, out):
    os.environ['TL_RUN_DIR']=str(out)
    os.environ['TL_ALLOW_LOCAL_COMPUTE']='1' if args.allow_local_compute else '0'
    for attr,var in [('model','TL_MODEL_DIR'),('sft','TL_SFT_ADAPTER'),('prepared','TL_PREPARED_DIR'),
                     ('f2','TL_F2_ADAPTER'),('benchmark','TL_BENCHMARK'),('dpo_run','TL_DPO_RUN'),('source_parquet','TL_SOURCE_PARQUET')]:
        if hasattr(args,attr):os.environ[var]=str(Path(getattr(args,attr)).expanduser().resolve())

def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.command in ['train-supervised','train-dpo','evaluate']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    require_compute(a.allow_local_compute)
    if a.command in ['stage-inputs','prepare-halueval','freeze-benchmark']:
        from . import prepare
        return getattr(prepare,{'stage-inputs':'stage','prepare-halueval':'halueval','freeze-benchmark':'benchmark'}[a.command])(a) or 0
    inputs=[getattr(a,k) for k in ['model','sft','split','prepared','f2','benchmark','dpo_run','source_parquet','records'] if hasattr(a,k)]
    for n in inputs:
        if not Path(n).exists():p.error('Required input does not exist: '+n)
    if hasattr(a,'model'):
        from .runtime import verify_model
        verify_model(a.model)
    if a.command in ['train-supervised','train-dpo','evaluate']:
        from .runtime import require_gpu
        require_gpu(a.allow_local_compute)
    out=claim_output(a.output,inputs);environment(a,out)
    if a.command=='prepare-dpo':
        from .dpo import dataset
        dataset.main();return 0
    if a.command=='train-supervised':
        from . import supervised
        supervised.configure(a.model,a.split,out)
        if a.recipe=='sft':supervised.run_lora_training(smoke=False)
        else:
            from . import f2_lite
            f2_lite.main(['--mode','full'])
        return 0
    if a.command in ['train-dpo','evaluate']:
        files=[*Path(a.model).glob('*.json'),*Path(a.model).glob('*.safetensors'),Path(a.sft)/'adapter_model.safetensors',Path(a.sft)/'adapter_config.json',
               Path(a.prepared)/'DATASET_MANIFEST.json']
        files+=list((Path(__file__).parent/'dpo').glob('*.py'))
        if a.command=='evaluate':files += [Path(a.f2)/'adapter_model.safetensors',Path(a.f2)/'adapter_config.json',Path(a.benchmark)]
        bind_files(out,files)
        if a.command=='train-dpo':
            from .dpo import train
            return train.full()
        from .dpo import evaluate
        return evaluate.main()
    if a.command=='summarize':
        from .metrics import ARMS,read_records,summarize
        rr={n:read_records(Path(a.records)/(n+'.jsonl')) for n in ARMS}
        dump(out/'summary.json',summarize(rr));return 0
    raise RuntimeError('Unhandled command')
