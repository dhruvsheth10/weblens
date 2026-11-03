"""Bounded public modern-generator development supplement; no paid generation."""
import hashlib,io,json,random,time,urllib.parse,urllib.request
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/modern-reference-development-2026-10-14/data';REPORT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
REPO='zr-zhang/MLLM-Generated-Image-Detection-Dataset';REV='651a325f0ece2744c21d6c68280d2aaac227177b'
STYLES=['Hybrid Images','Structure Images','Texture Images'];GENERATORS=['Real','GPT Image 2','Nano Banana 2']
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'objects').mkdir(exist_ok=True)
 reference=json.loads((ROOT/'backend/data/manifest.json').read_text());reserved={r['sourceGroup'] for r in reference if r['split']!='train'};all_existing={r['sourceGroup'] for r in reference}
 ids=[i for i in range(1,3001) if str(i) not in all_existing];random.Random(20261014).shuffle(ids);ids=ids[:24]
 index_path=REPORT/'modern-reference-file-index.json'
 if not index_path.exists():
  with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}/revision/{REV}?blobs=true',timeout=90) as response:index_path.write_bytes(response.read())
 entries=json.loads(index_path.read_text())['siblings']
 by_stem={str(Path(e['rfilename']).with_suffix('')):e for e in entries if e['rfilename'].startswith('images/')}
 plan={'repository':REPO,'revision':REV,'seed':20261014,'newSourceIds':ids,'styles':STYLES,'generators':GENERATORS,'split':'development only','excludedExistingReferenceGroups':sorted(all_existing),'byteCap':384*1024*1024,'perFileBytes':12*1024*1024,'minutes':15}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2));manifest=OUT/'manifest.jsonl';rows=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else [];seen={r['id'] for r in rows}
 # Reuse only the original train partition. Never open reference calibration/validation files.
 for r in reference:
  if r['split']!='train':continue
  key='existing:'+r['local']
  if key in seen:continue
  path=ROOT/'backend/data'/r['local'];data=path.read_bytes()
  if hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Changed train reference')
  record={'id':key,'split':'development','label':'ai' if r['label'] else 'real','generator':r['generator'],'style':r['style'],'sourceGroup':'mllm:'+r['sourceGroup'],'sourceHash':r['sha256'],'path':str(path),'basis':'publisher dataset label','repository':REPO,'revision':REV,'newDownload':False}
  rows.append(record);seen.add(key)
 used=sum(r.get('bytes',0) for r in rows if r['newDownload']);prior=REPORT/'modern-reference-collection.json';downloaded=json.loads(prior.read_text())['downloadBytes'] if prior.exists() else used;failures=[];started=time.monotonic();stop=False
 for source_id in ids:
  assert str(source_id) not in all_existing and str(source_id) not in reserved
  for style in STYLES:
   group=f'{source_id:04d}/{style}'
   if all('new:'+gen+'/'+group in seen for gen in GENERATORS):continue
   if time.monotonic()-started>plan['minutes']*60 or downloaded>=plan['byteCap']:stop=True;break
   triple=[]
   for generator in GENERATORS:
    stem=f'images/{generator}/{style}/{source_id:04d}';entry=by_stem.get(stem)
    if not entry:failures.append({'group':group,'generator':generator,'error':'Missing in publisher file index'});triple=[];break
    remote=entry['rfilename'];suffix=Path(remote).suffix.removeprefix('.');url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/'+urllib.parse.quote(remote,safe='/')
    try:
     with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'RedditLens-local-research/0.1'}),timeout=30) as response:
      declared=int(response.headers.get('Content-Length') or 0)
      if declared>plan['perFileBytes']:raise ValueError('File exceeds cap')
      data=response.read(min(plan['perFileBytes']+1,plan['byteCap']-downloaded+1));downloaded+=len(data)
     if not data or len(data)>plan['perFileBytes'] or downloaded>plan['byteCap']:raise ValueError('Download budget reached')
     expected=(entry.get('lfs') or {}).get('sha256')
     if expected and hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Publisher checksum mismatch')
     with Image.open(io.BytesIO(data)) as image:
      if image.width*image.height>40_000_000 or min(image.size)<128:raise ValueError('Unsupported geometry')
      image.verify()
     triple.append((generator,remote,data,suffix,url))
    except Exception as error:failures.append({'group':group,'generator':generator,'error':str(error)});triple=[];break
   if len(triple)!=3:continue
   for generator,remote,data,suffix,url in triple:
    sha=hashlib.sha256(data).hexdigest();path=OUT/'objects'/f'{sha}.{suffix}'
    if not path.exists():path.write_bytes(data)
    record={'id':'new:'+generator+'/'+group,'split':'development','label':'real' if generator=='Real' else 'ai','generator':generator,'style':style,'sourceGroup':'mllm:'+str(source_id),'sourceHash':sha,'path':str(path),'basis':'publisher dataset label','repository':REPO,'revision':REV,'url':url,'bytes':len(data),'newDownload':True};rows.append(record);seen.add(record['id']);used+=len(data)
   manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'references':len(rows),'newImages':sum(r['newDownload'] for r in rows),'downloadBytes':downloaded}),flush=True)
  if stop:break
 manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 from collections import Counter
 summary={'references':len(rows),'newImages':sum(r['newDownload'] for r in rows),'downloadBytes':downloaded,'storedNewBytes':used,'labels':dict(Counter(r['label'] for r in rows)),'generators':dict(Counter(r['generator'] for r in rows)),'styles':dict(Counter(r['style'] for r in rows)),'failures':failures,'plan':plan,'note':'Prepared for future development training; not inserted into the already-frozen student pilot. Publisher labels are provenance evidence, not independently audited origin. Same ID groups stay together; reserved existing reference groups were excluded.'}
 (REPORT/'modern-reference-collection.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['plan','failures']}),flush=True)
if __name__=='__main__':main()
