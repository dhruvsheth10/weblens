"""Dependency-free validation for local, versioned Reddit research candidates."""
import hashlib,json,math,re
from pathlib import PurePosixPath
from .semantic_profile import CHECKPOINT_SHA
from .mirror_profile import COMMUNITY_SHA,profile as mirror_profile

STUDENT_LAYOUTS={'community-vit-s224-supervised-last2-v1':(('artifacts','reddit-adaptation-2026-10-14','models'),'student'),
 'community-vit-s224-full-finetune-v2':(('artifacts','reddit-student-v2-2026-10-16','models'),'sherlock'),
 'community-vit-s224-full-finetune-sherlock2':(('artifacts','sherlock-2-2026-10-17','models'),'sherlock')}
def digest(value):return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='modelHash'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def sha(x):return isinstance(x,str) and bool(re.fullmatch('[a-f0-9]{64}',x))
def validate_profile(p):
 if not isinstance(p,dict) or p.get('schemaVersion')!=1 or p.get('kind') not in {'reddit-fusion','reddit-student'}:raise ValueError('Unsupported Reddit research profile')
 if p.get('modelHash')!=digest(p) or p.get('threshold')!=.5:raise ValueError('Research profile checksum or operating point mismatch')
 if not isinstance(p.get('model'),str) or not re.fullmatch('reddit-[a-z0-9._-]{1,65}',p['model']):raise ValueError('Invalid research identity')
 if p['kind']=='reddit-fusion':
  if p.get('pipeline')!='clipped-component-logits-standardized-logistic-v1' or p.get('components')!=['community_forensics','ferretnet_b','mirror'] or p.get('selectedFeatures') not in [[0,2],[0,1,2]] or p.get('clip')!=20.:raise ValueError('Unsupported fusion representation')
  weights=p.get('weight');indices=p['selectedFeatures']
  if not all(isinstance(i,int) and not isinstance(i,bool) for i in indices):raise ValueError('Invalid feature indices')
  if not isinstance(weights,list) or len(weights)!=len(indices) or not all(finite(v) for v in weights) or not finite(p.get('bias')):raise ValueError('Invalid fusion coefficients')
  if p.get('mirrorHash')!=mirror_profile()['modelHash'] or p.get('communityHash')!=COMMUNITY_SHA:raise ValueError('Fusion component identity mismatch')
 else:
  layout=STUDENT_LAYOUTS.get(p.get('pipeline'))
  if layout is None or p.get('sourceCheckpointSha256')!=CHECKPOINT_SHA or not sha(p.get('checkpointSha256')) or not sha(p.get('trainingHash')) or not finite(p.get('logitOffset')):raise ValueError('Invalid student checkpoint or pipeline')
  if not isinstance(p.get('checkpointPath'),str):raise ValueError('Invalid checkpoint path')
  path=PurePosixPath(p['checkpointPath']);folder,prefix=layout
  if path.is_absolute() or '..' in path.parts or path.parts[:3]!=folder or path.name!=f"{prefix}-{p['checkpointSha256']}.safetensors":raise ValueError('Student path must be a pinned project-local content-addressed weight file')
 return p
