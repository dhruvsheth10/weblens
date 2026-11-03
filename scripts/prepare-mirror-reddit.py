"""Freeze Reddit groups and a category-balanced tuning sample before inference."""
import collections,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'datasets/reddit/data';out=ROOT/'artifacts/mirror-evaluation-2026-10-13'
rows=[json.loads(x) for x in (data/'manifest.jsonl').read_text().splitlines()]
parent=list(range(len(rows)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):parent[find(i)]=find(j)
seen={}
for i,r in enumerate(rows):
 for key in [('post',r['postId']),('author',r['group']),*[('sha',r[k]['sha256']) for k in ('preview','original') if r.get(k) and r[k].get('sha256')]]:
  if key in seen:union(i,seen[key])
  else:seen[key]=i
# Conservative perceptual grouping prevents recrops/reposts with nearly identical dHashes crossing splits.
for i,r in enumerate(rows):
 for j in range(i):
  if (int(r['dhash'])^int(rows[j]['dhash'])).bit_count()<=4:union(i,j)
groups=collections.defaultdict(list)
for i,r in enumerate(rows):groups[find(i)].append(r)
manifest=[];conflicts=0
for group in groups.values():
 gid=hashlib.sha256('|'.join(sorted(r['id'] for r in group)).encode()).hexdigest()
 value=int(gid[:8],16)/0xffffffff
 split='development' if value<.7 else 'tuning' if value<.85 else 'holdout'
 mixed=len({r['label'] for r in group})>1
 if mixed:conflicts+=len(group);split='quarantine'
 for r in group:
  note=r.get('basisNote','').lower()
  ambiguous=r['basis']=='realorai-verified' and any(x in note for x in ['one is ai','other is real','mixed','partially','combination of'])
  manifest.append({**r,'split':'quarantine' if ambiguous else split,'sourceGroup':gid})
manifest.sort(key=lambda r:r['id'])
def save(name,x):
 target=out/name
 if name in {'reddit-splits.json','reddit/manifest.json'} and target.exists() and json.loads(target.read_text())!=x:
  raise ValueError('Frozen Reddit boundary would change. Use a new explicitly versioned collection instead: '+name)
 target.write_text(json.dumps(x,indent=2)+'\n')
save('reddit-splits.json',{'schemaVersion':1,'seed':'sha256 group identifiers v1','grouping':'post + author + exact hash + dHash distance <=4 connected components','note':'Holdout is reserved and not scored. Source labels are weak evidence, not independently verified ground truth.','rows':manifest})
counts=lambda rs:dict(collections.Counter(f"{r['label']}/{r['category']}" for r in rs))
source={s:{'labels':dict(collections.Counter(r['label'] for r in rows if r['source']==s)),'n':sum(r['source']==s for r in rows),'originals':sum(r['source']==s and bool((r.get('original') or {}).get('path')) for r in rows)} for s in sorted({r['source'] for r in rows})}
selected=[]
for cat in sorted({(r['label'],r['category']) for r in manifest}):
 candidates=[r for r in manifest if r['split']=='tuning' and (r['label'],r['category'])==cat]
 candidates.sort(key=lambda r:hashlib.sha256(('sample-v1'+r['id']).encode()).hexdigest())
 selected.extend(candidates[:50])
# Only this predetermined sample's pixel objects are opened. Reserved holdout is never scored.
bench=out/'reddit/calibration';bench.mkdir(parents=True,exist_ok=True);benchrows=[]
for r in selected:
 asset=r['preview'];relative=Path(asset['path'])
 if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe object path')
 src=data/relative
 if src.is_symlink() or not src.resolve().is_relative_to(data.resolve()):raise ValueError('Unsafe object path')
 content=src.read_bytes()
 if hashlib.sha256(content).hexdigest()!=asset['sha256']:raise ValueError('Changed Reddit object')
 file=Path(r['label'])/r['category']/(r['id']+src.suffix);target=bench/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
 benchrows.append({'file':'calibration/'+str(file),'label':r['label'],'group':r['category'],'sourceHash':asset['sha256'],'experimentSplit':'tuning','sourceGroup':r['sourceGroup'],'redditId':r['id']})
save('reddit/manifest.json',{'rows':benchrows})
spec={'root':str(bench),'manifest':str(out/'reddit/manifest.json'),'manifestSha256':hashlib.sha256((out/'reddit/manifest.json').read_bytes()).hexdigest()}
save('external-config.json',{'reddit-mirror':spec})
save('reddit-audit.json',{'total':len(rows),'labels':dict(collections.Counter(r['label'] for r in rows)),'categories':counts(rows),'originals':sum(bool((r.get('original') or {}).get('path')) for r in rows),'previewOnly':sum(not (r.get('original') or {}).get('path') for r in rows),'sources':source,'splits':{s:counts([r for r in manifest if r['split']==s]) for s in ['development','tuning','holdout','quarantine']},'conflictingGroupRows':conflicts,'sample':counts(selected),'sampleN':len(selected),'limitations':['Subreddit rules and generation communities are noisy labels.','RealOrAI answer parsing can label a whole gallery even when individual panels differ; manual review still required.','Source/category/age are correlated with class. No scraped data used for fitting in this experiment.']})
print(json.dumps({'splits':dict(collections.Counter(r['split'] for r in manifest)),'sample':counts(selected),'n':len(selected)},indent=2))
