"""Benchmark-only research engines. They are not enabled in the production picker."""
import os,time
import numpy as np
import timm,torch
from pathlib import Path
from safetensors.torch import load_file
from torchvision.transforms import v2 as T
from .assets import sha256
from .image_io import decode_image,sigmoid
from .reddit_research_profile import validate_profile

class RedditFusionEngine:
 def __init__(self,profile,combination=None):
  from .mirror import MirrorCommunityEngine
  self.profile=validate_profile(profile)
  if profile['kind']!='reddit-fusion':raise ValueError('Expected fusion profile')
  self.combination=combination or MirrorCommunityEngine();self.device=self.combination.device
  self.weight=np.asarray(profile['weight'],dtype=float);self.bias=profile['bias']
 def predict(self,data,variant='original'):
  result=self.combination.predict(data,variant);p=self.profile
  components=result['components'];values=np.array([components[key]['logit'] for key in p['components']]);values=np.clip(values,-p['clip'],p['clip'])[p['selectedFeatures']]
  logit=float(values@self.weight+self.bias)
  return {**result,'model':p['model'],'modelHash':p['modelHash'],'pipeline':p['pipeline'],'score':sigmoid(logit),'logit':logit,
    'decisionRule':'Development-fitted numeric fusion; operating point selected on Reddit tuning; research only'}

class RedditStudentEngine:
 def __init__(self,profile):
  self.profile=validate_profile(profile)
  if profile['kind']!='reddit-student':raise ValueError('Expected student profile')
  root=Path(__file__).resolve().parents[1];path=root/profile['checkpointPath']
  if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size>120*1024*1024 or sha256(path)!=profile['checkpointSha256']:raise ValueError('Student weight file missing, unsafe or changed')
  for ancestor in path.parents:
   if ancestor==root:break
   if ancestor.is_symlink():raise ValueError('Symlinked student path')
  requested=os.environ.get('LENS_DEVICE','auto');device=('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
  if device not in {'cpu','mps','cuda'}:raise ValueError('Invalid device')
  self.device=torch.device(device);torch.set_num_threads(min(4,os.cpu_count() or 1))
  self.model=timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k',pretrained=False);self.model.head=torch.nn.Linear(384,1)
  self.model.load_state_dict(load_file(str(path)),strict=True);self.model.requires_grad_(False).eval().to(self.device)
  self.transform=T.Compose([T.ToImage(),T.Resize(256),T.CenterCrop(224),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 @torch.inference_mode()
 def predict(self,data,variant='original'):
  started=time.perf_counter();image,size=decode_image(data,variant);pixels=self.transform(image).unsqueeze(0).to(self.device);raw=float(self.model(pixels).float().cpu()[0,0]);logit=raw-self.profile['logitOffset'];p=self.profile
  return {'model':p['model'],'modelHash':p['modelHash'],'pipeline':p['pipeline'],'score':sigmoid(logit),'logit':logit,'width':size[0],'height':size[1],
   'elapsedMs':round((time.perf_counter()-started)*1000),'engine':self.device.type,'variant':variant,'watermarkVerified':False,
   'components':{'reddit_student':{'score':sigmoid(raw),'logit':raw,'weight':1.,'logitOffset':p['logitOffset'],'crop':[224,224]}}}
