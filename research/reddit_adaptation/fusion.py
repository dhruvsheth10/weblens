"""Fit fusion on grouped Reddit development only; select operating points on tuning."""
import csv,math
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from .common import ROOT,OUT,load,write,digest,summarize,cutoff_at_category_fpr
FEATURES=['community_forensics','ferretnet_b','mirror'];CS=[.01,.1,1.,10.]

def weighted(rows):
 counts=Counter((r['label'],r['source'],r['sourceGroup']) for r in rows)
 sources={label:len({r['source'] for r in rows if r['label']==label}) for label in ['real','ai']}
 groups=Counter()
 for label,source,group in counts:groups[(label,source)]+=1
 w=np.array([1/(sources[r['label']]*groups[(r['label'],r['source'])]*counts[(r['label'],r['source'],r['sourceGroup'])]) for r in rows]);return w*len(w)/w.sum()

def fit(x,y,rows,c):
 w=weighted(rows);scaler=StandardScaler().fit(x,sample_weight=w)
 model=LogisticRegression(C=c,max_iter=2000,random_state=20261014).fit(scaler.transform(x),y,sample_weight=w)
 weight=model.coef_[0]/scaler.scale_;bias=float(model.intercept_[0]-weight@scaler.mean_)
 return weight,bias

def assemble():
 mapping={(r['dataset'],r['file']):r for r in load(OUT/'image-map.json')['rows']};audit={r['id']:r for r in load(OUT/'label-audit.json')['rows']};result={}
 for run in load(OUT/'mcp-runs.json'):
  if load(__import__('pathlib').Path(run['artifacts'])/'status.json')['status']!='complete':raise ValueError('Wait for every MCP run')
  rows=[];vectors=[];timing=[]
  for line in (__import__('pathlib').Path(run['artifacts'])/'scores.jsonl').read_text().splitlines():
   prediction=__import__('json').loads(line);r=mapping[(run['dataset'],prediction['file'])]
   rows.append({**r,'eligibleForWeakSupervision':audit[r['id']]['eligibleForWeakSupervision']});vectors.append([prediction['components'][k]['logit'] for k in FEATURES]);timing.append(prediction['timingMs'])
  result[run['dataset']]={'rows':rows,'x':np.asarray(vectors),'timing':timing}
 return result

def run():
 datasets=assemble();dev=datasets['reddit-development-preview'];tune=datasets['reddit-tuning-preview'];eligible=np.array([r['eligibleForWeakSupervision'] for r in dev['rows']]);devrows=[r for r in dev['rows'] if r['eligibleForWeakSupervision']]
 x=np.clip(dev['x'][eligible],-20,20);y=np.array([r['label']=='ai' for r in devrows],dtype=int);groups=np.array([r['sourceGroup'] for r in devrows])
 plan={'training':'frozen Reddit development only; ambiguous hard-case galleries excluded','features':FEATURES,'clipLogits':[-20,20],'C':CS,'folds':5,'seed':20261014,'selection':'grouped development OOF standardized partial AUROC at 1% FPR','operatingPoint':'maximum category-constrained tuning logit cut at empirical FPR <=1% in each real category','imageMapHash':digest(load(OUT/'image-map.json')),'tuningLabels':'source-derived, retained unchanged','holdoutAccess':False}
 old=OUT/'fusion-plan.json'
 if old.exists() and load(old)!=plan:raise ValueError('Frozen fusion plan changed')
 write(old,plan);candidates=[]
 splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261014).split(x,y,groups))
 for indices,name in [([0,2],'semantic-mirror'),([0,1,2],'semantic-ferret-mirror')]:
  for c in CS:
   oof=np.zeros(len(y))
   for train,test in splits:
    assert not set(groups[train])&set(groups[test]);w,b=fit(x[train][:,indices],y[train],[devrows[i] for i in train],c);oof[test]=x[test][:,indices]@w+b
   scores=summarize(oof,devrows);candidates.append({'features':indices,'name':name,'C':c,'oof':scores,'logits':oof})
 best=max(candidates,key=lambda r:(r['oof']['partialAuRoc01'],-len(r['features']),-r['C']))
 indices=best['features'];w,b=fit(x[:,indices],y,devrows,best['C'])
 tune_logits=np.clip(tune['x'],-20,20)[:,indices]@w+b
 cut=cutoff_at_category_fpr(tune_logits,[r['label']=='ai' for r in tune['rows']],[r['category'] for r in tune['rows']])
 profile={'schemaVersion':1,'kind':'reddit-fusion','model':'reddit-logit-fusion-v1','pipeline':'clipped-component-logits-standardized-logistic-v1','components':FEATURES,'selectedFeatures':indices,'clip':20.,'weight':w.tolist(),'bias':b-cut,'rawBias':b,'tuningCutLogit':cut,'threshold':.5,'training':{**plan,'selectedC':best['C'],'selectedFeatureSet':best['name']},'mirrorHash':load(ROOT/'backend/models/mirror-profile.json')['modelHash'],'communityHash':load(ROOT/'backend/models/community-ferret-profile.json')['modelHash']}
 profile['modelHash']=digest(profile);write(OUT/'fusion-profile.json',profile)
 results=[];ferretcut=math.log(.8304235935211183/(1-.8304235935211183))
 for dataset,v in datasets.items():
  r=v['rows'];raw=v['x'];base=np.maximum(raw[:,0],raw[:,1]-ferretcut)
  systems={'Community + Ferret':base,'MIRROR':raw[:,2],'OR blend':np.maximum(base,raw[:,2]),'Mean logits':(base+raw[:,2])/2,'Development-fitted fusion':np.clip(raw,-20,20)[:,indices]@w+b-cut}
  # Each baseline also gets its own tuning-selected category-constrained cut.
  for name,logits in systems.items():
   if name=='Development-fitted fusion':op=0.
   else:
    tx=tune['x'];tb=np.maximum(tx[:,0],tx[:,1]-ferretcut);tl={'Community + Ferret':tb,'MIRROR':tx[:,2],'OR blend':np.maximum(tb,tx[:,2]),'Mean logits':(tb+tx[:,2])/2}[name]
    op=cutoff_at_category_fpr(tl,[z['label']=='ai' for z in tune['rows']],[z['category'] for z in tune['rows']])
   results.append({'dataset':dataset,'system':name,'policy':'frozen0.5' if name!='Development-fitted fusion' else 'tuning-category-fpr01','cutLogit':0.,'metrics':summarize(logits,r)})
   results.append({'dataset':dataset,'system':name,'policy':'tuning-category-fpr01','cutLogit':op,'metrics':summarize(logits,r,op)})
 write(OUT/'fusion-results.json',{'selected':{k:v for k,v in best.items() if k!='logits'},'candidates':[{k:v for k,v in c.items() if k!='logits'} for c in candidates],'results':results,'warning':'Development fusion scores are in-sample after fitting; use OOF for development model selection. Tuning selects thresholds, so it is not independent validation.'})
 np.savez_compressed(OUT/'fusion-dev-oof.npz',logits=best['logits'],ids=np.array([r['id'] for r in devrows]))
 for r in results:
  if r['dataset']=='reddit-tuning-preview':print(r['system'],r['policy'],{k:round(r['metrics'][k],4) for k in ['precision','recall','f1','fpr']},flush=True)
if __name__=='__main__':run()
