import hashlib,json,os,re,sys,unicodedata
from pathlib import Path
sys.dont_write_bytecode=True
# Runtime locations are explicit CLI inputs; never point into historical runs.
HERE=Path(os.environ.get('TL_RUN_DIR','outputs/unconfigured')).resolve()
ROOT=HERE
WORK=HERE.parent
BASE=Path(os.environ.get('TL_MODEL_DIR','data/staged/model')).resolve()
SFT=Path(os.environ.get('TL_SFT_ADAPTER','outputs/unconfigured-sft')).resolve()
F2=Path(os.environ.get('TL_F2_ADAPTER','outputs/unconfigured-f2')).resolve()
PREPARED=Path(os.environ.get('TL_PREPARED_DIR',str(HERE))).resolve()
DPO_RUN=Path(os.environ.get('TL_DPO_RUN',str(HERE))).resolve()
SOURCE_PARQUET=Path(os.environ.get('TL_SOURCE_PARQUET','data/staged/ultrafeedback/data/train-00000-of-00001.parquet')).resolve()
BENCH=Path(os.environ.get('TL_BENCHMARK','data/benchmark/truthfulqa_817_frozen.jsonl')).resolve()
REVISION='770076f077c4c5e298498fa32f804857f46d5134'
REPO='argilla/ultrafeedback-binarized-preferences-cleaned'
PARQUET_SHA='27dc59041d73654aecfa7685b6f839a0bbb2014d640ee92eefa047b262d1f1ca'
def sha(s):return hashlib.sha256(s if isinstance(s,bytes) else s.encode()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def rank(s):return sha('UF-V1|42|'+s)
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def words(s):return norm(s).split()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False,sort_keys=True)
def jl(p,rr):
 with Path(p).open('x') as f:
  for r in rr:f.write(canon(r)+'\n')
def read(p):
 # JSON permits Unicode paragraph separators inside a string. splitlines()
 # wrongly treats them as record delimiters; JSONL is framed by physical LF.
 with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]
def slurm():
 from ..runtime import require_compute
 require_compute(os.environ.get('TL_ALLOW_LOCAL_COMPUTE')=='1')
def prior_core():
 from . import core
 return core
