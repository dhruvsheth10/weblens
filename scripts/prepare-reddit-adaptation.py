"""Publish explicit dev/tuning-only MCP datasets without changing frozen splits."""
import collections,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14';DATA=ROOT/'datasets/reddit/data'
SOURCE=ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
 text=json.dumps(value,indent=2)+'\n'
 if path.exists() and path.read_text()!=text:raise ValueError('Frozen preparation changed: '+str(path))
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def main():
 OUT.mkdir(parents=True,exist_ok=True);source=json.loads(SOURCE.read_text());registry={};mapping=[];coverage={}
 for split in ['development','tuning']:
  selected=[r for r in source['rows'] if r['split']==split]
  for role in ['preview','original']:
   name=f'reddit-{split}-{role}';base=OUT/'datasets'/name;root=base/'calibration';records=[]
   for r in selected:
    asset=r.get(role) or {}
    if not asset.get('path'):continue
    relative=Path(asset['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe asset')
    src=DATA/relative
    if src.is_symlink() or not src.resolve().is_relative_to(DATA.resolve()):raise ValueError('Unsafe asset')
    if digest(src)!=asset['sha256']:raise ValueError('Changed image: '+r['id'])
    file=Path(r['label'])/r['category']/(r['id']+src.suffix);dst=root/file;dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)  # immutable bytes, no redundant multi-GB copies
    if digest(dst)!=asset['sha256']:raise ValueError('Changed staged image')
    records.append({'file':'calibration/'+str(file),'label':r['label'],'group':r['category'],'sourceHash':asset['sha256'],
       'experimentSplit':'tuning','usageSplit':split,'sourceGroup':r['sourceGroup'],'redditId':r['id']})
    mapping.append({'dataset':name,'file':str(file),'usageSplit':split,'role':role,'sourceHash':asset['sha256'],
       'id':r['id'],'postId':r['postId'],'sourceGroup':r['sourceGroup'],'label':r['label'],'category':r['category'],
       'source':r['source'],'basis':r['basis'],'basisNote':r.get('basisNote',''),'preview':r['preview'],'original':r.get('original')})
   manifest=base/'manifest.json';write(manifest,{'note':'MCP external transport uses tuning; usageSplit retains the actual frozen development/tuning boundary. This dataset never contains holdout or quarantine images.','rows':records})
   registry[name]={'root':str(root),'manifest':str(manifest),'manifestSha256':digest(manifest)}
   coverage[name]={'n':len(records),'categories':dict(collections.Counter(x['label']+'/'+x['group'] for x in records))}
 write(OUT/'external-config.json',registry)
 write(OUT/'image-map.json',{'sourceManifestSha256':digest(SOURCE),'rows':mapping})
 write(OUT/'data-coverage.json',{'sourceManifestSha256':digest(SOURCE),'datasets':coverage,
   'limits':['Source-derived labels remain weak evidence.','External MCP partition=tuning is a transport constraint; fit only usageSplit=development.','Original coverage is source-confounded; preview is the primary evaluation input.','Reserved holdout and calibration validation/audit pixels are excluded.']})
 print(json.dumps(coverage,indent=2))
if __name__=='__main__':main()
