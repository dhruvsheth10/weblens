import hashlib,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def load(path):return json.loads(Path(path).read_text())
def sigmoid(x):return np.exp(-np.logaddexp(0,-np.asarray(x,dtype=float)))
def safe_data(row):
 path=Path(row['path']);root=Path(row['root'])
 if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Image escaped explicit root')
 data=path.read_bytes()
 if hashlib.sha256(data).hexdigest()!=row['sourceHash']:raise ValueError('Changed image')
 return data

def cutoff_at_category_fpr(logits,labels,categories,cap=.01):
 if not isinstance(cap,(int,float)) or not math.isfinite(cap) or not 0<=cap<1:raise ValueError('FPR cap must be in [0,1)')
 logits=np.asarray(logits,dtype=float);labels=np.asarray(labels,dtype=int);categories=np.asarray(categories)
 if not (len(logits)==len(labels)==len(categories)):raise ValueError('Mismatched category inputs')
 if not np.isfinite(logits).all():raise ValueError('Nonfinite logit')
 # Small logit margin protects strict cuts against CPU/GPU and folded-bias rounding.
 cuts=[]
 for category in sorted(set(categories[labels==0])):
  values=np.sort(logits[(labels==0)&(categories==category)])[::-1]
  allowed=int(math.floor(len(values)*cap));cuts.append(float(np.nextafter(values[allowed],np.inf))+1e-4)
 if not cuts:raise ValueError('No negative category to constrain')
 return max(cuts)

def wilson(success,n):
 if not n:return None
 z=1.959963984540054;p=success/n;den=1+z*z/n;middle=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,middle-radius),min(1,middle+radius)]

def summarize(logits,rows,cutoff=0.):
 logits=np.asarray(logits,dtype=float);y=np.array([r['label']=='ai' for r in rows]);flags=logits>=cutoff
 tp,fp,tn,fn=[int(v.sum()) for v in [y&flags,~y&flags,~y&~flags,y&~flags]]
 recall=tp/(tp+fn) if tp+fn else 0;fpr=fp/(fp+tn) if fp+tn else 0
 categories={}
 for category in sorted({r['category'] for r in rows}):
  mask=np.array([r['category']==category for r in rows]);negative=mask&~y;positive=mask&y
  categories[category]={'n':int(mask.sum()),'realN':int(negative.sum()),'aiN':int(positive.sum()),'fp':int((negative&flags).sum()),'tp':int((positive&flags).sum()),
   'fpr':float(flags[negative].mean()) if negative.any() else None,'recall':float(flags[positive].mean()) if positive.any() else None,
   'fprWilson95':wilson(int((negative&flags).sum()),int(negative.sum()))}
 prevalence={str(p):recall*p/(recall*p+fpr*(1-p)) if recall*p+fpr*(1-p) else None for p in [.01,.05,.1]}
 return {'n':len(rows),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'fpr':fpr,
  'auroc':float(roc_auc_score(y,logits)) if len(set(y))==2 else None,'partialAuRoc01':float(roc_auc_score(y,logits,max_fpr=.01)) if len(set(y))==2 else None,
  'averagePrecision':float(average_precision_score(y,logits)) if y.any() else None,'precisionAtAssumedPrevalence':prevalence,'categories':categories,
  'precisionWilson95':wilson(tp,tp+fp),'recallWilson95':wilson(tp,tp+fn),'fprWilson95':wilson(fp,fp+tn),
  'uncertaintyNote':'Wilson intervals treat images as independent and may be optimistic for correlated source groups. Source-label errors are not captured.'}
