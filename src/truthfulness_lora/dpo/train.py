"""SFT-initialized FP32 DPO with complete atomic boundary checkpoints.

Old validated serialization and DPO objective are reused. Per-branch sums replace
CUDA index_add to make controlled process-restart comparison deterministic.
"""
import argparse,copy,hashlib,json,math,os,random,signal,sys,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from .common import *
CONFIG=dict(seed=42,epochs=3,beta=.1,learning_rate=5e-6,accumulation=4,max_length=2048,
            validation_interval=500,patience=3,min_delta=1e-4,gradient_clip=1.,rank=16,alpha=32,
            precision='float32',attention='eager',scheduler='constant')
STOP=False
def stopped(signum,frame):
 global STOP;STOP=True
def setup():
 slurm();assert torch.cuda.is_available();torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 signal.signal(signal.SIGUSR1,stopped);signal.signal(signal.SIGTERM,stopped)
def fsync_dir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic_json(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fsync_dir(path.parent)
def parameter_hash(model):
 h=hashlib.sha256()
 for n,p in model.named_parameters():
  h.update(n.encode());arr=p.detach().cpu().contiguous().numpy();h.update(memoryview(arr).cast('B'))
 return h.hexdigest()
def capture_rng():
 return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(rng):
 random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['cpu'])
 if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
def optimizer_for(model):
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
 return opt,torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda _:1.)
def checkpoint(root,model,opt,scheduler,state,isbest=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);tag=f'u{state["updates"]:07d}-e{state["epoch"]}-c{state["cursor"]}-'+uuid.uuid4().hex[:10]
 tmp=root/('.tmp-'+tag);final=root/tag;tmp.mkdir()
 assert all(p.grad is None for p in model.parameters()),'Checkpoint only at optimizer/accumulation boundary'
 if isbest:state['best_checkpoint']=str(final)
 model.save_pretrained(tmp,safe_serialization=True)
 payload=dict(config=CONFIG,state=copy.deepcopy(state),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),
              accumulation_boundary=True,code_sha256=digest(Path(__file__)))
 torch.save(payload,tmp/'training_state.pt')
 for p in tmp.iterdir():
  if p.is_file():
   with p.open('rb') as f:os.fsync(f.fileno())
 dump(tmp/'CHECKPOINT_MANIFEST.json',dict(files={p.name:digest(p) for p in tmp.iterdir() if p.is_file()},
      updates=state['updates'],epoch=state['epoch'],cursor=state['cursor'],accumulation_boundary=True))
 with (tmp/'CHECKPOINT_MANIFEST.json').open('rb') as f:os.fsync(f.fileno())
 fsync_dir(tmp);os.rename(tmp,final);fsync_dir(root)
 atomic_json(root/'LATEST.json',dict(path=str(final),manifest_sha256=digest(final/'CHECKPOINT_MANIFEST.json')))
 return final
def restore(path,model,opt,scheduler):
 from safetensors.torch import load_file
 from peft import set_peft_model_state_dict
 path=Path(path);manifest=json.loads((path/'CHECKPOINT_MANIFEST.json').read_text())
 for n,h in manifest['files'].items():assert digest(path/n)==h,(path,n)
 payload=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
 assert payload['config']==CONFIG and payload['accumulation_boundary']
 assert payload['code_sha256']==digest(Path(__file__)),'Resume code changed'
 set_peft_model_state_dict(model,load_file(str(path/'adapter_model.safetensors')))
 opt.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
 restore_rng(payload['rng']);return payload['state']
def batch(pair,pad,device):
 n=max(len(f['input_ids']) for f in pair);ids=torch.full((len(pair),n),pad,device=device,dtype=torch.long);att=torch.zeros_like(ids);mask=torch.zeros_like(ids,dtype=torch.bool)
 for i,f in enumerate(pair):
  m=len(f['input_ids']);ids[i,:m]=torch.tensor(f['input_ids'],device=device);att[i,:m]=1;mask[i,:m]=torch.tensor(f['mask'],device=device)
 return dict(input_ids=ids,attention_mask=att,mask=mask)
def score(model,features,pad,device='cuda'):
 b=batch(features,pad,device);logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
 values=[]
 for i in range(len(features)):
  mask=b['mask'][i,1:] & b['attention_mask'][i,1:].bool()
  selected=logits[i,:-1][mask].float();targets=b['input_ids'][i,1:][mask]
  values.append(F.log_softmax(selected,dim=-1).gather(1,targets[:,None]).squeeze(1).sum())
 out=torch.stack(values);assert torch.isfinite(out).all();return out
def serialize(tok,q,answer):
 messages=[dict(role='user',content='Answer truthfully and concisely.\n\nQuestion: '+q)]
 prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 full=tok.apply_chat_template(messages+[dict(role='assistant',content=answer)],tokenize=False,add_generation_prompt=False)
 assert full==prompt+'<think>\n\n</think>\n\n'+answer+'<|im_end|>\n'
 ids=tok(full,add_special_tokens=False)['input_ids'];prefix=tok(prompt,add_special_tokens=False)['input_ids']
 assert ids[:len(prefix)]==prefix and len(ids)<=CONFIG['max_length']
 stops=[i for i in range(len(prefix),len(ids)) if ids[i] in [151645,151643]]
 assert len(stops)==1 and ids[stops[0]]==151645 and tok.decode(ids[stops[0]+1:])=='\n'
 return dict(input_ids=ids,mask=[len(prefix)<=i<=stops[0] for i in range(len(ids))])
def features(rows,tok,core):
 result={}
 for r in rows:
  result[r['source_row_id']]=[serialize(tok,r['prompt'],r[k]) for k in ['chosen_response','rejected_response']]
 return result
def update(policy,opt,sched,window,feat,ref,pad,core,device='cuda'):
 opt.zero_grad(set_to_none=True);policy.train();losses=[]
 for key in window:
  p=score(policy,feat[key],pad,device);loss=core.dpo_loss(p,torch.tensor(ref[key],device=device),CONFIG['beta'])
  assert torch.isfinite(loss);(loss/len(window)).backward();losses.append(float(loss.detach()))
 norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],CONFIG['gradient_clip'],error_if_nonfinite=True)
 assert torch.isfinite(norm) and norm>0
 opt.step();sched.step();opt.zero_grad(set_to_none=True)
 return dict(loss=sum(losses)/len(losses),grad_norm=float(norm),pairs=len(window))
def validate(model,keys,feat,ref,pad,core,device='cuda'):
 model.eval();losses=[];margins=[]
 with torch.inference_mode():
  for k in keys:
   if STOP:raise InterruptedError('Validation interrupted at a record boundary')
   pp=score(model,feat[k],pad,device);losses.append(float(core.dpo_loss(pp,torch.tensor(ref[k],device=device),CONFIG['beta'])));margins.append(float(pp[0]-pp[1]))
 return dict(loss=float(np.mean(losses)),ranking_accuracy=float(np.mean([1 if m>1e-6 else .5 if abs(m)<=1e-6 else 0 for m in margins])),pairs=len(keys))
def load_data():
 manifest=json.loads((PREPARED/'DATASET_MANIFEST.json').read_text())
 for n,h in manifest['files'].items():assert digest(PREPARED/n)==h,n
 train=read(PREPARED/'dataset/train.jsonl');val=read(PREPARED/'dataset/validation.jsonl')
 assert not {r['group_id'] for r in train}&{r['group_id'] for r in val}
 return manifest,train,val
def identity():
 from ..runtime import verify_bindings
 return verify_bindings(HERE)
def ref_cache(reference,feat,pad,out,reference_hash):
 out=Path(out);out.mkdir(exist_ok=True);cache={};meta=out/'LATEST.json'
 if meta.exists():
  info=json.loads(meta.read_text());p=Path(info['path']);assert digest(p)==info['sha256']
  blob=torch.load(p,map_location='cpu',weights_only=False);assert blob['reference_hash']==reference_hash
  assert blob['dataset_sha256']==digest(PREPARED/'DATASET_MANIFEST.json');cache=blob['scores']
 def save():
  target=out/f'scores-{len(cache):05d}.pt'
  if target.exists():
   prior=torch.load(target,map_location='cpu',weights_only=False)
   assert prior['reference_hash']==reference_hash and prior['dataset_sha256']==digest(PREPARED/'DATASET_MANIFEST.json') and prior['scores']==cache
   atomic_json(meta,dict(path=str(target),sha256=digest(target)));return
  tmp=target.with_suffix('.tmp');torch.save(dict(reference_hash=reference_hash,dataset_sha256=digest(PREPARED/'DATASET_MANIFEST.json'),scores=cache),tmp)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  os.rename(tmp,target);fsync_dir(out);atomic_json(meta,dict(path=str(target),sha256=digest(target)))
 reference.eval()
 with torch.inference_mode():
  for k in feat:
   if k not in cache:cache[k]=score(reference,feat[k],pad).cpu().tolist()
   if len(cache)%128==0:save()
   if STOP:save();raise InterruptedError('Reference scoring checkpointed')
 save();return cache

def full():
 setup();identity()  # Explicit CLI input binding; no fictional scientific GO gate.
 out=HERE/'full_run';out.mkdir(mode=0o700,exist_ok=True)
 manifest,train,val=load_data();ids=identity();core=prior_core()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True);feat=features(train+val,tok,core)
 reference=core.load_effective(SFT,'cuda');policy=core.load_effective(SFT,'cuda',True);core.assert_isolated(policy,reference)
 rh=parameter_hash(reference);opt,sched=optimizer_for(policy);started=time.monotonic()
 state=dict(epoch=0,cursor=0,updates=0,order=[],best_val=float('inf'),best_checkpoint=None,bad=0,last_validation=-1,
            dataset_manifest_sha256=digest(PREPARED/'DATASET_MANIFEST.json'),reference_hash=rh)
 pointer=out/'checkpoints/LATEST.json'
 if pointer.exists():state=restore(json.loads(pointer.read_text())['path'],policy,opt,sched)
 assert state['dataset_manifest_sha256']==digest(PREPARED/'DATASET_MANIFEST.json') and state['reference_hash']==rh
 try:ref=ref_cache(reference,feat,tok.pad_token_id,out/'reference_cache',rh)
 except InterruptedError:return 75
 keys=[r['source_row_id'] for r in train];vk=[r['source_row_id'] for r in val]
 if not (out/'run_manifest.json').exists():dump(out/'run_manifest.json',dict(config=CONFIG,identities=ids,dataset_manifest_sha256=state['dataset_manifest_sha256'],gpu=torch.cuda.get_device_name(0),reference_hash=rh))
 def validation_checkpoint():
  try:result=validate(policy,vk,feat,ref,tok.pad_token_id,core)
  except InterruptedError:
   checkpoint(out/'checkpoints',policy,opt,sched,state);raise
  improved=result['loss']<state['best_val']-CONFIG['min_delta']
  if improved:state['best_val']=result['loss'];state['bad']=0
  else:state['bad']+=1
  state['last_validation']=state['updates'];path=checkpoint(out/'checkpoints',policy,opt,sched,state,improved)
  with (out/'validation.jsonl').open('a') as f:f.write(canon(dict(update=state['updates'],epoch=state['epoch'],checkpoint=str(path),improved=improved,**result))+'\n');f.flush();os.fsync(f.fileno())
  return result
 if state['updates']==0 and state['last_validation']<0:
  initial=validation_checkpoint();assert abs(initial['loss']-math.log(2))<1e-3
 elif state['updates']>state['last_validation'] and (state['updates']%CONFIG['validation_interval']==0 or state['cursor']==len(keys)):
  validation_checkpoint()
 reason='validation_patience' if state['bad']>=CONFIG['patience'] else 'epochs_completed'
 while state['epoch']<CONFIG['epochs'] and state['bad']<CONFIG['patience']:
  if not state['order']:
   order=list(keys);random.Random(CONFIG['seed']+state['epoch']).shuffle(order);state['order']=order;state['cursor']=0
  while state['cursor']<len(keys):
   if STOP:checkpoint(out/'checkpoints',policy,opt,sched,state);return 75
   start=state['cursor'];window=state['order'][start:start+CONFIG['accumulation']]
   result=update(policy,opt,sched,window,feat,ref,tok.pad_token_id,core)
   state['cursor']+=len(window);state['updates']+=1
   assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
   with (out/'training.jsonl').open('a') as f:f.write(canon(dict(update=state['updates'],epoch=state['epoch'],cursor=state['cursor'],elapsed_seconds=time.monotonic()-started,**result))+'\n')
   if STOP:checkpoint(out/'checkpoints',policy,opt,sched,state);return 75
   if state['updates']%CONFIG['validation_interval']==0 or state['cursor']==len(keys):validation_checkpoint()
   if STOP:checkpoint(out/'checkpoints',policy,opt,sched,state);return 75
   if state['bad']>=CONFIG['patience']:reason='validation_patience';break
  if state['bad']>=CONFIG['patience']:break
  state['epoch']+=1;state['cursor']=0;state['order']=[]
  checkpoint(out/'checkpoints',policy,opt,sched,state)
 assert parameter_hash(reference)==rh and state['best_checkpoint']
 for n,h in ids.items():assert digest(ROOT/n)==h,n
 atomic_json(out/'selected_checkpoint.json',dict(path=state['best_checkpoint'],metric='validation_DPO_loss_only',best_validation_loss=state['best_val'],baseline_eligible=True))
 atomic_json(out/'completion.json',dict(status='SUCCESS',reason=reason,updates=state['updates'],epochs_completed=state['epoch'],reference_unchanged=True,
             selected_checkpoint=state['best_checkpoint'],elapsed_seconds=time.monotonic()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
 return 0
if __name__=='__main__':
 try:sys.exit(full())
 except InterruptedError:sys.exit(75)
