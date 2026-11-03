"""Verify hashes and approximate alignment of newly recovered dev-only pairs."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'datasets/reddit/pairs-development-2026-10-14';OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
def dhash(image):
 a=np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16);return int(''.join('1' if b else '0' for b in (a[:,1:]>a[:,:-1]).ravel()),2)
def main():
 rows=[]
 for line in (BASE/'manifest.jsonl').read_text().splitlines():
  r=json.loads(line);assert r['split']=='development'
  paths=[(ROOT/'datasets/reddit/data'/r['preview']['path'],r['preview']),(BASE/r['original']['path'],r['original'])];values=[]
  for path,asset in paths:
   assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256']
   with Image.open(path) as image:values.append((dhash(image),image.width/image.height))
  distance=(values[0][0]^values[1][0]).bit_count();aspect=abs(values[0][1]/values[1][1]-1)
  rows.append({'id':r['id'],'label':r['label'],'category':r['category'],'dhashDistance':distance,'relativeAspectDifference':aspect,'alignmentReviewRequired':distance>12 or aspect>.05})
 summary={'pairs':len(rows),'hashChecks':len(rows)*2,'alignmentReviewRequired':sum(r['alignmentReviewRequired'] for r in rows),'categories':dict(collections.Counter(r['label']+'/'+r['category'] for r in rows)),
 'note':'dHash/aspect are alignment diagnostics, not proof of image origin. Crop differences require review before clean-to-preview distillation.','rows':rows}
 (OUT/'pair-integrity.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='rows'})
if __name__=='__main__':main()
