"""Numerical evidence on explicit development inputs; no accuracy/threshold fitting."""
import json,os,time,psutil,torch,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.mirror import MirrorEngine,MirrorCommunityEngine
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/mirror-evaluation-2026-10-13'
rows=json.loads((ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json').read_text())['rows']
chosen=[]
for label in ['real','ai']:
 chosen.append(next(r for r in rows if r['experimentSplit']=='development' and r['label']==label))
os.environ['LENS_DEVICE']='mps';engine=MirrorEngine();blend=MirrorCommunityEngine(engine)
evidence=[]
for r in chosen:
 relative=Path(r['file']);assert relative.parts[0]=='calibration'
 data=(Path('/Users/dhruvsheth/Downloads/benchmark/calibration')/Path(*relative.parts[1:])).read_bytes()
 for variant in ['original','reddit-webp640']:
  m=engine.predict(data,variant);b=blend.predict(data,variant);c=blend.community.predict(data,variant)
  assert abs(b['logit']-max(m['logit'],c['logit']))<1e-6
  evidence.append({'file':r['file'],'sourceHash':r['sourceHash'],'variant':variant,'mps':m,'blend':b,'community':c})
memory={'rssBytes':psutil.Process().memory_info().rss,'mpsAllocatedBytes':torch.mps.current_allocated_memory(),'mpsDriverBytes':torch.mps.driver_allocated_memory(),'systemRamBytes':psutil.virtual_memory().total}
engine.model.to('cpu');engine.device=torch.device('cpu')
for r in evidence:
 relative=Path(r['file']);data=(Path('/Users/dhruvsheth/Downloads/benchmark/calibration')/Path(*relative.parts[1:])).read_bytes();cpu=engine.predict(data,r['variant'])
 r['cpu']=cpu;r['logitDifference']=abs(cpu['logit']-r['mps']['logit'])
 assert r['logitDifference']<.001,r
out.joinpath('numerical-parity.json').write_text(json.dumps({'memoryWithBlend':memory,'loadMs':engine.load_ms,'parameters':engine.parameter_count,'examples':evidence},indent=2))
print(json.dumps({'memory':memory,'maxLogitDifference':max(r['logitDifference'] for r in evidence)}),flush=True)
