import sys,json,time,types,torch,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.mirror import MirrorEngine,CachedMemoryBank
from backend.vendor.mirror.upstream import MirrorMemoryBank
root=Path(__file__).resolve().parents[1];e=MirrorEngine();bank=e.model.memory_bank
rows=json.loads((root/'artifacts/mirror-evaluation-2026-10-13/reddit/manifest.json').read_text())['rows']
out=[]
for row in rows[:3]:
 data=(root/'artifacts/mirror-evaluation-2026-10-13/reddit'/row['file']).read_bytes()
 results={}
 for mode,fn in [('uncached',MirrorMemoryBank.forward),('cached',CachedMemoryBank.forward)]:
  bank.forward=types.MethodType(fn,bank)
  predictions=[e.predict(data) for _ in range(4)]
  results[mode]={'score':predictions[-1]['score'],'logit':predictions[-1]['logit'],'warmMs':[r['elapsedMs'] for r in predictions[1:]]}
 assert results['cached']['logit']==results['uncached']['logit'],results
 out.append({'file':row['file'],**results})
(root/'artifacts/mirror-evaluation-2026-10-13/cache-parity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
