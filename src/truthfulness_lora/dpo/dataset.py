"""Pinned-source CPU preparation, full benchmark screens and seed-42 grouping."""
import collections,json,math,os,re,sys,time
sys.dont_write_bytecode=True
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from sklearn.feature_extraction.text import TfidfVectorizer
from .common import *

def domain(q):
 for pat,label in [(r'\b(python|javascript|java|c\+\+|code|sql|programming|algorithm|html|software|function)\b','programming'),
  (r'\b(calculate|equation|probability|algebra|mathematics|arithmetic|solve|integral|geometry)\b','mathematics'),
  (r'\b(physics|chemical|biology|science|scientific|quantum|molecule|experiment|dna|planet)\b','science'),
  (r'\b(story|poem|poetry|fiction|creative|lyrics|novel|character)\b','creative_writing'),
  (r'\b(history|historical|war|century|president|ancient|empire)\b','history_society'),
  (r'\b(health|medical|medicine|disease|doctor|symptom|nutrition|diet)\b','health'),
  (r'\b(business|market|finance|economic|company|sales|investment)\b','business')]:
  if re.search(pat,q,re.I):return label
 return 'general_instruction'

class Serializer:
 def __init__(self):
  self.tok=Tokenizer.from_file(str(BASE/'tokenizer.json'))
  self.template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string(json.loads((BASE/'tokenizer_config.json').read_text())['chat_template'])
 def pair(self,q,a,b):
  prompt=self.template.render(messages=[dict(role='user',content='Answer truthfully and concisely.\n\nQuestion: '+q)],add_generation_prompt=True)
  prefix=self.tok.encode(prompt,add_special_tokens=False).ids;result=[]
  for answer in [a,b]:
   full=self.template.render(messages=[dict(role='user',content='Answer truthfully and concisely.\n\nQuestion: '+q),dict(role='assistant',content=answer)],add_generation_prompt=False)
   assert full==prompt+'<think>\n\n</think>\n\n'+answer+'<|im_end|>\n'
   ids=self.tok.encode(full,add_special_tokens=False).ids
   assert ids[:len(prefix)]==prefix and ids[len(prefix):].count(151645)==1
   result.append(dict(full_tokens=len(ids),answer_tokens=len(self.tok.encode(answer,add_special_tokens=False).ids),
                      words=len(answer.split()),characters=len(answer)))
  return result

def grouping(rows):
 parent=list(range(len(rows)));owner={};spans=collections.defaultdict(list)
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for i,r in enumerate(rows):
  sigs=['q:'+norm(r['prompt'])]
  for key in ['chosen_response','rejected_response']:
   if len(words(r[key]))>=20:sigs.append('a:'+norm(r[key]))
  for sig in sigs:
   if sig in owner:union(i,owner[sig])
   else:owner[sig]=i
  ww=words(r['prompt'])
  for k in range(len(ww)-19):spans[sha(' '.join(ww[k:k+20]))[:24]].append(i)
 ignored=0
 for indices in spans.values():
  uu=set(indices)
  if len(uu)>100:ignored+=1;continue
  if len(uu)>1:
   first=min(uu)
   for j in uu:union(first,j)
 groups=collections.defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 gids={i:sha(canon(sorted(rows[j]['source_content_sha256'] for j in indices))) for indices in groups.values() for i in indices}
 return gids,dict(groups=len(groups),largest_group=max(map(len,groups.values())),common20word_spans_ignored=ignored)

def contamination(rows,out):
 bench=read(BENCH);assert len(bench)==817
 qnorm=[norm(x['question']) for x in bench];anchor=collections.defaultdict(list);shindex=collections.defaultdict(set);qsets=[]
 short={q:i for i,q in enumerate(qnorm) if len(q.split())<6}
 short_pattern=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(q) for q in sorted(short,key=len,reverse=True))+r')(?!\w)')
 for i,q in enumerate(qnorm):
  ww=q.split();k=min(8,len(ww))
  if k>=6:anchor[(k,tuple(ww[:k]))].append(('question',i,q))
  ss={tuple(ww[j:j+5]) for j in range(len(ww)-4)};qsets.append(ss)
  for sh in ss:shindex[sh].add(i)
  for name in ['mc1_targets','mc2_targets']:
   for choice in bench[i][name]['choices']:
    cc=norm(choice);cw=cc.split()
    if len(cw)>=8 and len(cc)>=40:anchor[(8,tuple(cw[:8]))].append(('long_choice',i,cc))
 reasons=collections.defaultdict(set);hits=[]
 for r in rows:
  idx=r['source_index']
  for field in ['prompt','chosen_response','rejected_response']:
   text=norm(r[field]);ww=text.split();found=set()
   for hit in short_pattern.finditer(text):found.add(('question',short[hit.group()]))
   for k in [6,7,8]:
    if len(ww)<k:continue
    for j in range(len(ww)-k+1):
     for typ,bi,target in anchor.get((k,tuple(ww[j:j+k])),[]):
      if target in text:found.add((typ,bi))
   for typ,bi in sorted(found):reasons[idx].add(typ);hits.append(dict(index=idx,benchmark_index=bi,kind=typ,field=field))
  ww=words(r['prompt']);ss={tuple(ww[j:j+5]) for j in range(len(ww)-4)}
  candidates={bi for sh in ss for bi in shindex.get(sh,[])}
  for bi in candidates:
   qq=qsets[bi]
   if not qq:continue
   containment=len(ss&qq)/len(qq);jac=len(ss&qq)/len(ss|qq)
   if containment>=.8 or jac>=.65:
    reasons[idx].add('question_5shingle_near');hits.append(dict(index=idx,benchmark_index=bi,kind='question_5shingle_near',containment=containment,jaccard=jac))
 # Two lexical representations, independently fit only for contamination search.
 print('Exact/choice/shingle source scan complete',flush=True)
 nearest=np.zeros(len(rows));which=np.zeros(len(rows),dtype=int);kind=['']*len(rows)
 prompts=[r['prompt'] for r in rows];bq=[x['question'] for x in bench]
 for label,vec in [('word',TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=100000,sublinear_tf=True,dtype=np.float32)),
                   ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=50000,min_df=2,sublinear_tf=True,dtype=np.float32))]:
  xx=vec.fit_transform(prompts+bq);train=xx[:len(rows)];bv=xx[len(rows):]
  print(f'{label} contamination TFIDF fitted {xx.shape}',flush=True)
  for start in range(0,len(rows),512):
   scores=(train[start:start+512]@bv.T).toarray();ix=scores.argmax(axis=1);maxes=scores[np.arange(len(ix)),ix]
   for off,(bi,score) in enumerate(zip(ix,maxes)):
    i=start+off
    if score>nearest[i]:nearest[i]=score;which[i]=bi;kind[i]=label
   del scores
  del xx,train,bv
 for i,score in enumerate(nearest):
  if score>=.65:reasons[i].add('question_tfidf_near');hits.append(dict(index=i,benchmark_index=int(which[i]),kind='question_tfidf_near',score=float(score),representation=kind[i]))
 remaining=sorted((i for i in range(len(rows)) if not reasons.get(i)),key=lambda i:-nearest[i])[:40]
 jl(out/'benchmark_top_remaining_40.jsonl',[dict(source_id=rows[i]['source_row_id'],prompt=rows[i]['prompt'],benchmark_question=bq[which[i]],score=float(nearest[i]),representation=kind[i]) for i in remaining])
 jl(out/'benchmark_hits.jsonl',hits)
 dump(out/'benchmark_screen.json',dict(benchmark_rows=817,benchmark_sha256=digest(BENCH),rows_flagged=len(reasons),
      reason_counts=dict(collections.Counter(s for rr in reasons.values() for s in rr)),
      caveat='Lexical exact/5shingle/two TFIDF searches across complete source; no guarantee of semantic decontamination. Short generic choices are not unique contamination anchors.'))
 return {i:rr for i,rr in reasons.items() if rr}

def quotas(counts,n):
 assert 0<=n<=sum(counts.values()),'Quota cannot exceed available rows'
 if n==0:return {k:0 for k in counts}
 total=sum(counts.values());q={k:min(counts[k],int(n*counts[k]/total)) for k in counts}
 while sum(q.values())<n:
  k=max((k for k in counts if q[k]<counts[k]),key=lambda k:(n*counts[k]/total-q[k],k));q[k]+=1
 return q
def sample(pool,n):
 if n==0:return []
 by=collections.defaultdict(list)
 for r in pool:by[r['source']].append(r)
 quota=quotas({k:len(v) for k,v in by.items()},n);selected=[]
 for source,rr in sorted(by.items()):
  bb=collections.defaultdict(list)
  for r in sorted(rr,key=lambda r:rank(r['source_content_sha256'])):bb[r['token_direction']].append(r)
  q=quota[source];tie=min(len(bb['tie']),round(q*len(bb['tie'])/len(rr)));left=q-tie
  short=min(len(bb['chosen_shorter']),left//2);long=min(len(bb['chosen_longer']),left-short)
  remain=left-short-long
  if remain:
   extra=min(remain,len(bb['chosen_shorter'])-short);short+=extra;remain-=extra
  if remain:long+=remain
  selected +=bb['tie'][:tie]+bb['chosen_shorter'][:short]+bb['chosen_longer'][:long]
 assert len(selected)==n
 return selected



def main():
 slurm();start=time.monotonic();out=HERE/'dataset';out.mkdir(mode=0o700,exist_ok=False)
 assert digest(SOURCE_PARQUET)==PARQUET_SHA
 raw=pq.read_table(SOURCE_PARQUET).to_pylist();assert len(raw)==60917
 ser=Serializer();rows=[]
 for i,r in enumerate(raw):
  reason=[];prompt=r.get('prompt');a=r.get('chosen');b=r.get('rejected')
  valid=isinstance(prompt,str) and isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==2
  if valid:
   valid=all(isinstance(x,dict) and set(x)=={'role','content'} for x in a+b) and [x['role'] for x in a]==[x['role'] for x in b]==['user','assistant']
  if valid:valid=a[0]['content']==b[0]['content']==prompt and all(isinstance(x['content'],str) for x in a+b)
  if not valid:reason.append('unsupported_schema_or_prompt_mismatch')
  aa=a[-1]['content'] if valid else '';bb=b[-1]['content'] if valid else '';prompt=prompt if isinstance(prompt,str) else ''
  if not all(x.strip() for x in [prompt,aa,bb]):reason.append('empty_field')
  if any(t in prompt+aa+bb for t in ['<|im_start|>','<|im_end|>','<|endoftext|>','<think>','</think>','\x00']) or re.search(r'<\|[^>]+\|>',prompt+aa+bb):reason.append('embedded_control_tokens')
  if norm(aa)==norm(bb):reason.append('identical_answers')
  cr=r.get('chosen-rating');br=r.get('rejected-rating')
  ratings=all(isinstance(v,(int,float)) and math.isfinite(v) and 1<=v<=5 for v in [cr,br])
  if not ratings:reason.append('invalid_ratings')
  elif cr<4 or cr-br<.5:reason.append('rating_quality_or_gap')
  if not all(isinstance(r.get(k),str) and r[k].strip() for k in ['source','chosen-model','rejected-model']):reason.append('missing_provenance')
  lens=None
  if aa.startswith('\n') or bb.startswith('\n'):reason.append('native_template_would_remove_leading_newlines')
  if valid and not any(x in reason for x in ['empty_field','embedded_control_tokens','native_template_would_remove_leading_newlines']):
   lens=ser.pair(prompt,aa,bb)
   if max(x['full_tokens'] for x in lens)>2048:reason.append('complete_serialized_branch_over2048')
   if min(x['answer_tokens'] for x in lens)==0 or min(x['words'] for x in lens)==0:reason.append('empty_encoded_answer')
   elif not .5<=lens[1]['answer_tokens']/lens[0]['answer_tokens']<=2 or not .5<=lens[1]['words']/lens[0]['words']<=2:reason.append('broad_ratio_outside_half_to_two')
  direction='tie' if not lens or lens[0]['answer_tokens']==lens[1]['answer_tokens'] else 'chosen_shorter' if lens[0]['answer_tokens']<lens[1]['answer_tokens'] else 'chosen_longer'
  rows.append(dict(source_index=i,source_row_id=f'uf:{i}',source_content_sha256=sha(canon(r)),source=r.get('source','unknown'),
      prompt=prompt,chosen_response=aa,rejected_response=bb,chosen_rating=cr,rejected_rating=br,chosen_model=r.get('chosen-model'),rejected_model=r.get('rejected-model'),
      domain_heuristic=domain(prompt),lengths=lens,token_direction=direction,exclusions=reason))
  if i%10000==0:print(f'schema/token lengths {i}/60917',flush=True)
 del raw
 gid,groupstats=grouping(rows);print('Full-source grouping complete',flush=True)
 benchmark=contamination(rows,out);bg={gid[i] for i in benchmark}
 for r in rows:
  r['group_id']=gid[r['source_index']]
  if r['group_id'] in bg:r['exclusions'].append('benchmark_connected_group')
 eligible=[r for r in rows if not r['exclusions']];representatives={}
 for r in sorted(eligible,key=lambda r:rank(r['source_content_sha256'])):representatives.setdefault(r['group_id'],r)
 unique=list(representatives.values())
 for r in eligible:
  if representatives[r['group_id']]['source_index']!=r['source_index']:r['exclusions'].append('within_source_duplicate_group')
 total=min(10000,len(unique));nv=min(total,500,max(1,round(total*.05)));nt=total-nv
 val=sample(unique,nv);valgroups={r['group_id'] for r in val};train=sample([r for r in unique if r['group_id'] not in valgroups],nt)
 assert not {r['group_id'] for r in train}&valgroups and not any(r['exclusions'] for r in train+val)
 for split,rr in [('train',train),('validation',val)]:
  for r in rr:r['split']=split
  jl(out/(split+'.jsonl'),sorted(rr,key=lambda r:rank(split+'|'+r['source_content_sha256'])))
 selectedids={r['source_index'] for r in train+val}
 jl(out/'all60917_ledger.jsonl',[{k:v for k,v in r.items() if k not in ['prompt','chosen_response','rejected_response']}|dict(selected=r['source_index'] in selectedids) for r in rows])
 excluded=[r for r in rows if r['exclusions'] and r['chosen_response'] and r['rejected_response']]
 # Reviewer packet generation deliberately omitted in the portable release adapter.
 dump(out/'audit_summary.json',dict(source_rows=60917,eligible_pair_rows=len(eligible),unique_eligible_groups=len(unique),
      train=len(train),validation=len(val),exclusion_reasons=dict(collections.Counter(x for r in rows for x in r['exclusions'])),
      source_distribution=dict(collections.Counter(r['source'] for r in rows)),eligible_source_distribution=dict(collections.Counter(r['source'] for r in unique)),
      selected_source_distribution=dict(collections.Counter(r['source'] for r in train+val)),
      selected_domain_distribution=dict(collections.Counter(r['domain_heuristic'] for r in train+val)),
      grouping=groupstats,elapsed_seconds=time.monotonic()-start))
 dump(HERE/'DATASET_MANIFEST.json',dict(source_repo=REPO,source_revision=REVISION,parquet_sha256=PARQUET_SHA,
      source_rows=60917,train=len(train),validation=len(val),seed=42,max_sequence_length=2048,
      preference_labels='GPT4 multidimensional average-quality ratings; not independent factual-truthfulness labels',human_verified=0,
      files={str(p.relative_to(HERE)):digest(p) for p in out.glob('*') if p.is_file()},
      train_validation_group_overlap=0,selected_benchmark_group_hits=0,full_semantic_decontamination=False,
      dataset_change='SFT/F2-Lite HaluEval versus DPO UltraFeedback: complete-recipe comparison, not DPO-only causal effect'))
 print((out/'audit_summary.json').read_text())
if __name__=='__main__':main()
