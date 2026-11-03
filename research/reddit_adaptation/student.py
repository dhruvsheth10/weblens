"""Supervised low-budget student pilot. No teacher targets or tuning gradients."""
from __future__ import annotations
import hashlib,io,json,math,os,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision.transforms import functional as TF,InterpolationMode
from .common import ROOT,OUT,digest,load,write,safe_data,cutoff_at_category_fpr,summarize
from .fusion import weighted
from backend.community import CommunityBackbone
from backend.image_io import decode_image
from backend.semantic_profile import CHECKPOINT_SHA,canonical_hash,PIPELINE

PLAN={'seed':20261014,'backboneSha256':CHECKPOINT_SHA,'training':'eligible Reddit development previews plus authorized calibration development',
 'validationAccess':False,'holdoutAccess':False,'weakLabelPolicy':'source-derived Reddit labels, ambiguous hard-case galleries excluded; no pseudo-labels',
 'headC':[.01,.05,.2],'headSelection':'5-fold source-group OOF partial AUROC at 1% FPR',
 'pilot':{'epochs':2,'drawsPerEpoch':4096,'batch':8,'accumulation':2,'trainable':'last two transformer blocks, final norm and binary head','backboneLr':1e-5,'headLr':1e-4,'weightDecay':.05,'precision':'float32'},
 'augmentation':'class-independent no-upscale width choice native/384/640/960; 40% unchanged, 30% JPEG q60..96, 30% WebP q60..95; short edge256 + random224 crop; no text reversal',
 'weighting':'equal domain/class; source/group weights capped at eight times within-domain/class median to limit rare noisy examples',
 'selection':'fixed two-epoch final checkpoint; tuning used only after training for thresholds/evaluation'}

def inputs():
 audit={r['id']:r for r in load(OUT/'label-audit.json')['rows']};rows=[]
 for r in load(OUT/'image-map.json')['rows']:
  if r['role']!='preview':continue
  rows.append({'id':'reddit:'+r['id'],'redditId':r['id'],'split':r['usageSplit'],'domain':'reddit','label':r['label'],'category':r['category'],
    'source':r['source'],'sourceGroup':'reddit:'+r['sourceGroup'],'sourceHash':r['sourceHash'],'root':str(ROOT/'datasets/reddit/data'),
    'path':str(ROOT/'datasets/reddit/data'/r['preview']['path']),'eligible':audit[r['id']]['eligibleForWeakSupervision']})
 manifest=load(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json')
 for r in manifest['rows']:
  if r['experimentSplit']!='development':continue
  relative=Path(r['file']);assert relative.parts[0]=='calibration'
  rows.append({'id':'calibration:'+r['file'],'split':'development','domain':'calibration','label':r['label'],'category':r['group'],
    'source':'calibration-'+r['label'],'sourceGroup':'calibration:'+r['sourceGroup'],'sourceHash':r['sourceHash'],
    'root':'/Users/dhruvsheth/Downloads/benchmark/calibration','path':str(Path('/Users/dhruvsheth/Downloads/benchmark/calibration')/Path(*relative.parts[1:])),'eligible':True})
 return rows

def weights(rows):
 out=np.zeros(len(rows))
 for domain in sorted({r['domain'] for r in rows}):
  ids=np.array([i for i,r in enumerate(rows) if r['domain']==domain]);raw=weighted([rows[i] for i in ids])
  for label in ['real','ai']:
   selected=np.array([j for j,i in enumerate(ids) if rows[i]['label']==label])
   if not len(selected):raise ValueError('Every training domain must contain both labels')
   values=raw[selected];values=np.minimum(values,8*np.median(values));out[ids[selected]]=values/values.sum()
 return out*len(out)/out.sum()


def extract(backbone,rows):
 identity=digest({'rows':rows,'checkpoint':CHECKPOINT_SHA,'transform':'community224-pinned-v1'});path=OUT/'student-features.npz'
 if path.exists():
  data=np.load(path,allow_pickle=False)
  if data['identity'].item()!=identity:raise ValueError('Feature identity changed')
  return data['features']
 # Existing development-only features include all four predeclared views.
 old=load(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json')['rows'];old=[r for r in old if r['experimentSplit']=='development']
 saved=np.load(ROOT/'artifacts/semantic-preview-2026-10-11/development-features.npz',allow_pickle=False)
 prior_plan=load(ROOT/'artifacts/semantic-preview-2026-10-11/plan.json')
 if saved['dataHash'].item()!=canonical_hash({'rows':old,'plan':prior_plan}):raise ValueError('Original feature cache identity mismatch')
 oldfeatures={r['file']:saved['features'][i] for i,r in enumerate(old)}
 features=[];started=time.perf_counter()
 for i,r in enumerate(rows):
  if r['domain']=='calibration':feature=oldfeatures[r['id'].removeprefix('calibration:')]
  else:
   feature=np.repeat(backbone.encode(decode_image(safe_data(r))[0])[None,:],4,axis=0)
  features.append(feature)
  if (i+1)%200==0:print('FEATURES',i+1,len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
 result=np.array(features);np.savez_compressed(path,features=result,identity=identity);return result

def fit_head(features,rows):
 y=np.array([r['label']=='ai' for r in rows],int);groups=np.array([r['sourceGroup'] for r in rows]);views=features.shape[1]
 def fit(index,c):
  w=weights([rows[i] for i in index]);x=features[index].reshape(-1,384);sample=np.repeat(w,views)/views
  scaler=StandardScaler().fit(x,sample_weight=sample);model=LogisticRegression(C=c,max_iter=2000,random_state=20261014).fit(scaler.transform(x),np.repeat(y[index],views),sample_weight=sample)
  weight=model.coef_[0]/scaler.scale_;bias=float(model.intercept_[0]-weight@scaler.mean_);return weight,bias
 candidates=[];splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261014).split(features,y,groups))
 for c in PLAN['headC']:
  oof=np.zeros(len(rows))
  for train,test in splits:
   assert not set(groups[train])&set(groups[test]);w,b=fit(train,c);oof[test]=(features[test]@w+b).mean(axis=1)
  metric=summarize(oof,rows);candidates.append({'C':c,'metrics':metric})
 best=max(candidates,key=lambda r:(r['metrics']['partialAuRoc01'],-r['C']));w,b=fit(np.arange(len(rows)),best['C'])
 write(OUT/'student-head-fit.json',{'selectedC':best['C'],'developmentOOF':candidates,'weight':w.tolist(),'bias':b,'plan':PLAN})
 return w,b

class TrainingImages(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  r=self.rows[index];image=decode_image(safe_data(r))[0]
  width=random.choice([image.width,384,640,960]);scale=min(1.,width/image.width)
  if scale<1:image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
  choice=random.random()
  if choice>=.4:
   stream=io.BytesIO()
   if choice<.7:image.save(stream,'JPEG',quality=random.randint(60,96))
   else:image.save(stream,'WEBP',quality=random.randint(60,95),method=2)
   with Image.open(io.BytesIO(stream.getvalue())) as encoded:image=encoded.convert('RGB')
  image=TF.resize(image,256,interpolation=InterpolationMode.BILINEAR,antialias=True)
  # Seeded random crop applied identically to both labels. Preserve text orientation.
  top=random.randint(0,image.height-224);left=random.randint(0,image.width-224);image=TF.crop(image,top,left,224,224)
  value=TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
  return value,float(r['label']=='ai')

def evaluate(backbone,head,rows):
 results=[]
 with torch.inference_mode():
  for i,r in enumerate(rows):
   data=safe_data(r);image=decode_image(data)[0];t=time.perf_counter();pixels=backbone.transform(image).unsqueeze(0).to(backbone.device);logit=float(backbone.model(pixels).float().cpu()[0,0]);ms=(time.perf_counter()-t)*1000
   results.append({**r,'logit':logit,'elapsedMs':ms})
   if (i+1)%200==0:print('EVALUATE',i+1,len(rows),flush=True)
 return results

def calibration_check(backbone,weight,bias,student_cut,head_cut):
 frozen=CommunityBackbone();rows=[r for r in load(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json')['rows'] if r['experimentSplit']=='tuning'];results=[];predictions=[]
 for variant in ['original','reddit-webp640']:
  student_logits=[];head_logits=[];metadata=[]
  for i,r in enumerate(rows):
   relative=Path(r['file']);assert relative.parts[0]=='calibration'
   root=Path('/Users/dhruvsheth/Downloads/benchmark/calibration');row={'path':str(root/Path(*relative.parts[1:])),'root':str(root),'sourceHash':r['sourceHash'],'label':r['label'],'category':r['group'],'file':r['file']}
   image=decode_image(safe_data(row),variant)[0]
   with torch.inference_mode():
    pixels=backbone.transform(image).unsqueeze(0).to(backbone.device);sl=float(backbone.model(pixels).float().cpu()[0,0])
   hl=float(frozen.encode(image)@weight+bias);student_logits.append(sl);head_logits.append(hl);metadata.append(row)
   predictions.append({'file':r['file'],'sourceHash':r['sourceHash'],'variant':variant,'studentLogit':sl,'headLogit':hl})
   if (i+1)%150==0:print('CALIBRATION TRANSFER',variant,i+1,len(rows),flush=True)
  results.append({'variant':variant,'student':summarize(student_logits,metadata,student_cut),'head':summarize(head_logits,metadata,head_cut),'note':'Both thresholds were selected on Reddit tuning, not this dataset.'})
 write(OUT/'student-calibration-transfer.json',{'results':results,'predictions':predictions})

def run():
 random.seed(PLAN['seed']);np.random.seed(PLAN['seed']);torch.manual_seed(PLAN['seed'])
 rows=inputs();identity=digest(rows);plan={**PLAN,'inputRowsHash':identity,'labelAuditHash':digest(load(OUT/'label-audit.json'))};planpath=OUT/'student-plan.json'
 if planpath.exists() and load(planpath)!=plan:raise ValueError('Predeclared student plan changed')
 write(planpath,plan);backbone=CommunityBackbone();features=extract(backbone,rows)
 indices=np.array([i for i,r in enumerate(rows) if r['split']=='development' and r['eligible']]);trainrows=[rows[i] for i in indices]
 w,b=fit_head(features[indices],trainrows)
 # Freeze a head-only candidate before any fine-tuning. The raw bias has no tuning cut.
 write(OUT/'student-head-raw.json',{'weight':w.tolist(),'bias':b,'trainingHash':digest(plan)})
 model=backbone.model
 with torch.no_grad():model.head.weight.copy_(torch.tensor(w,dtype=torch.float32)[None,:].to(backbone.device));model.head.bias.fill_(b)
 model.requires_grad_(False)
 for module in [model.blocks[-2:],model.norm,model.head]:module.requires_grad_(True)
 parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('head.')]
 optimizer=torch.optim.AdamW([{'params':parameters,'lr':PLAN['pilot']['backboneLr']},{'params':model.head.parameters(),'lr':PLAN['pilot']['headLr']}],weight_decay=PLAN['pilot']['weightDecay'])
 generator=torch.Generator().manual_seed(PLAN['seed']);sampler=WeightedRandomSampler(torch.tensor(weights(trainrows),dtype=torch.float64),PLAN['pilot']['drawsPerEpoch'],replacement=True,generator=generator)
 loader=DataLoader(TrainingImages(trainrows),batch_size=PLAN['pilot']['batch'],sampler=sampler,num_workers=0)
 optimizer.zero_grad(set_to_none=True);history=[];started=time.perf_counter();trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 print('TRAIN',len(trainrows),'images; trainable parameters',trainable,flush=True)
 for epoch in range(PLAN['pilot']['epochs']):
  model.eval() # Disable dropout; gradients remain enabled in the selected blocks.
  losses=[]
  for step,(pixels,labels) in enumerate(loader):
   pixels=pixels.to(backbone.device);labels=labels.float().to(backbone.device);logits=model(pixels).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
   accumulation=PLAN['pilot']['accumulation'];window_start=step-step%accumulation;window=min(accumulation,len(loader)-window_start)
   (loss/window).backward();losses.append(float(loss.detach().cpu()))
   if (step+1)%accumulation==0 or step+1==len(loader):
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
   if (step+1)%25==0:
    status={'epoch':epoch+1,'step':step+1,'steps':len(loader),'meanLoss':float(np.mean(losses)),'seconds':time.perf_counter()-started};write(OUT/'student-status.json',status);print(json.dumps(status),flush=True)
  history.append({'epoch':epoch+1,'meanLoss':float(np.mean(losses))})
 model.eval().requires_grad_(False);folder=OUT/'models';folder.mkdir(exist_ok=True);temporary=folder/'student.partial.safetensors'
 save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(temporary));sha=hashlib.sha256(temporary.read_bytes()).hexdigest();target=folder/f'student-{sha}.safetensors';temporary.replace(target)
 write(OUT/'student-training.json',{'plan':plan,'history':history,'trainN':len(trainrows),'trainableParameters':trainable,'totalParameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,'weightsSha256':sha,'weightsPath':str(target)})
 # Model selection is finished. Only now inspect tuning predictions and select thresholds.
 tuning=[r for r in rows if r['domain']=='reddit' and r['split']=='tuning'];predictions=evaluate(backbone,None,tuning)
 write(OUT/'student-tuning-predictions.json',{'rows':predictions})
 logits=np.array([r['logit'] for r in predictions]);cut=cutoff_at_category_fpr(logits,[r['label']=='ai' for r in tuning],[r['category'] for r in tuning]);write(OUT/'student-results.json',{'rawCut0':summarize(logits,tuning),'categoryFpr01':summarize(logits,tuning,cut),'cutLogit':cut,'warning':'Weak source labels and tuning-selected threshold; independent holdout was not evaluated.'})
 profile={'schemaVersion':1,'kind':'reddit-student','model':'reddit-student-last2-v1','pipeline':'community-vit-s224-supervised-last2-v1','checkpointSha256':sha,'sourceCheckpointSha256':CHECKPOINT_SHA,'checkpointPath':str(target.relative_to(ROOT)),'logitOffset':cut,'threshold':.5,'trainingHash':digest(plan)};profile['modelHash']=digest(profile);write(OUT/'student-profile.json',profile)
 # Head-only scores use the same fixed tuning features; no fitting on tuning.
 ti=[i for i,r in enumerate(rows) if r['domain']=='reddit' and r['split']=='tuning'];headlogits=features[ti,0]@w+b;hc=cutoff_at_category_fpr(headlogits,[r['label']=='ai' for r in tuning],[r['category'] for r in tuning])
 write(OUT/'student-head-results.json',{'rawCut0':summarize(headlogits,tuning),'categoryFpr01':summarize(headlogits,tuning,hc),'cutLogit':hc})
 headprofile={'schemaVersion':1,'kind':'semantic-head','model':'community-reddit-head-v1','pipeline':PIPELINE,'checkpointSha256':CHECKPOINT_SHA,'head':{'weight':w.tolist(),'bias':b-hc},'threshold':.5,'uncertainThreshold':.5,'training':{'planHash':digest(plan),'rawBias':b,'cutLogit':hc,'developmentOnly':True}}
 headprofile['modelHash']=canonical_hash(headprofile);write(OUT/'student-head-profile.json',headprofile)
 calibration_check(backbone,w,b,cut,hc)
 write(OUT/'student-status.json',{'status':'complete','seconds':time.perf_counter()-started})
 print('STUDENT COMPLETE',json.dumps(load(OUT/'student-results.json')['categoryFpr01']),flush=True)
if __name__=='__main__':run()
