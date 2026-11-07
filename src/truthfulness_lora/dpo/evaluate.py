"""Prepared four-arm paired MC scoring; runs only after successful new training."""
import gc,json,math,os,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
from .common import *
from . import train as tr

def metric(scores,target,kind):
 values=np.array([scores[x] for x in target['choices']],dtype=float);labels=np.array(target['labels'])
 if kind=='mc1':
  tied=np.isclose(values,values.max(),atol=1e-8,rtol=0)
  return float(labels[tied].sum()/tied.sum())
 masses=np.exp(values-values.max());return float(masses[labels==1].sum()/masses.sum())
def main():
 tr.setup();completion=json.loads((DPO_RUN/'full_run/completion.json').read_text());assert completion['status']=='SUCCESS'
 selected=Path(completion['selected_checkpoint']).resolve();assert selected.is_relative_to((DPO_RUN/'full_run/checkpoints').resolve())
 cm=json.loads((selected/'CHECKPOINT_MANIFEST.json').read_text())
 for n,h in cm['files'].items():assert digest(selected/n)==h
 frozen=json.loads((PREPARED/'dataset/benchmark_screen.json').read_text());assert digest(BENCH)==frozen['benchmark_sha256']
 bench=read(BENCH);assert len(bench)==817;core=prior_core();tr.identity()
 out=HERE/'evaluation';out.mkdir(mode=0o700,exist_ok=True)
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True)
 adapters={'Base':None,'SFT':SFT,'F2-Lite':F2,'DPO':selected};results={}
 identities={}
 for arm,adapter in adapters.items():
  path=out/(arm+'.jsonl');existing=read(path) if path.exists() else []
  assert all(x['benchmark_index']==i and x['question']==bench[i]['question'] for i,x in enumerate(existing))
  identities[arm]=dict(base_revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',adapter=str(adapter) if adapter else None,
   adapter_sha256=digest(adapter/'adapter_model.safetensors') if adapter else None)
  model=core.load_effective(adapter,'cuda') if adapter else AutoModelForCausalLM.from_pretrained(str(BASE),torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True).to('cuda').eval().requires_grad_(False)
  with torch.inference_mode(),path.open('a') as f:
   for i,item in enumerate(bench):
    if i<len(existing):continue
    answers=list(dict.fromkeys(item['mc1_targets']['choices']+item['mc2_targets']['choices']));scores={}
    for answer in answers:scores[answer]=float(tr.score(model,[tr.serialize(tok,item['question'],answer)],tok.pad_token_id)[0])
    record=dict(benchmark_index=i,question=item['question'],scores=scores,
     mc1=metric(scores,item['mc1_targets'],'mc1'),mc2=metric(scores,item['mc2_targets'],'mc2'))
    f.write(canon(record)+'\n');f.flush();os.fsync(f.fileno());existing.append(record)
    if tr.STOP:return 75
  results[arm]=existing;assert len(existing)==817
  del model;gc.collect();torch.cuda.empty_cache()
 rng=np.random.default_rng(42);indices=rng.integers(0,817,size=(10000,817));summary={};deltas={}
 for arm,rr in results.items():
  summary[arm]={}
  for metric_name in ['mc1','mc2']:
   values=np.array([r[metric_name] for r in rr]);summary[arm][metric_name]=dict(mean=float(values.mean()),bootstrap95=np.quantile(values[indices].mean(1),[.025,.975]).tolist())
 for other in ['Base','SFT','F2-Lite']:
  deltas['DPO-minus-'+other]={}
  for metric_name in ['mc1','mc2']:
   dd=np.array([a[metric_name]-b[metric_name] for a,b in zip(results['DPO'],results[other])])
   deltas['DPO-minus-'+other][metric_name]=dict(mean=float(dd.mean()),paired_bootstrap95=np.quantile(dd[indices].mean(1),[.025,.975]).tolist())
 dump(out/'RESULTS.json',dict(arms=summary,paired_differences=deltas,identities=identities,benchmark_sha256=digest(BENCH),questions=817,
  scoring='Question-only SFT chat instruction; raw summed EOS-inclusive B completion log-likelihood; MC1 tie credit; MC2 normalized true-answer probability mass',
  caveat='Complete recipes use different data. Conditional item bootstrap omits training-seed uncertainty. Historical six-shot scores excluded.'))
 return 0
if __name__=='__main__':sys.exit(main())
