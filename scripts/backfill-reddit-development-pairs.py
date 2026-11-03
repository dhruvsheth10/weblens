"""Recover balanced development originals without changing frozen manifests."""
import collections,hashlib,io,json,re,time,urllib.parse,urllib.request
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json';OUT=ROOT/'datasets/reddit/pairs-development-2026-10-14';REPORT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
EXT={'JPEG':'jpg','PNG':'png','WEBP':'webp','AVIF':'avif'};QUOTAS={'real/photo':40,'real/art':40,'real/meme':40,'real/screenshot':40,'ai/ai-art':40}

def candidates(row):
 result=[];known=(row.get('original') or {}).get('url')
 if known:
  u=urllib.parse.urlsplit(known)
  if u.scheme=='https' and u.netloc=='i.redd.it':result.append(known)
 u=urllib.parse.urlsplit(row['preview']['url']);name=u.path.rsplit('/',1)[-1];match=re.fullmatch(r'(?:.*-v0-)?([A-Za-z0-9]{6,24})\.(jpe?g|png|webp)',name,re.I)
 if not match:return result
 image_id,ext=match.groups();extensions=[ext.lower(),urllib.parse.parse_qs(u.query).get('format',[''])[0],'jpg','png']
 for suffix in extensions:
  if suffix not in {'jpg','jpeg','png','webp'}:continue
  url=f'https://i.redd.it/{image_id}.{suffix}'
  if url not in result:result.append(url)
 return result[:3]

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'objects').mkdir(exist_ok=True)
 plan={'sourceManifestSha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'scope':'development only; metadata selection independent of model predictions','quotas':QUOTAS,'maxBytes':512*1024*1024,'perImageBytes':12*1024*1024,'maxMinutes':15}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2));manifest=OUT/'manifest.jsonl';saved=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else [];seen={r['id'] for r in saved};counts=collections.Counter(r['label']+'/'+r['category'] for r in saved);groups=collections.Counter(r['sourceGroup'] for r in saved);used=sum(r['original']['bytes'] for r in saved)
 rows=[r for r in json.loads(SOURCE.read_text())['rows'] if r['split']=='development' and r['label']+'/'+r['category'] in QUOTAS and not (r.get('original') or {}).get('path')]
 rows.sort(key=lambda r:hashlib.sha256(('balanced-pairs-v1'+r['id']).encode()).hexdigest());failures=[];start=time.monotonic()
 # Interleave categories so a time/byte cap cannot spend everything on one source type.
 queues={key:[r for r in rows if r['label']+'/'+r['category']==key] for key in QUOTAS}
 ordered=[]
 for i in range(max(map(len,queues.values()))):
  ordered.extend(q[i] for q in queues.values() if i<len(q))
 for row in ordered:
  key=row['label']+'/'+row['category']
  if row['id'] in seen or counts[key]>=QUOTAS[key] or groups[row['sourceGroup']]>=2:continue
  if time.monotonic()-start>plan['maxMinutes']*60 or used>=plan['maxBytes']:break
  errors=[];data=None
  for url in candidates(row):
   try:
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'image/avif,image/webp,image/png,image/jpeg,*/*;q=0.5'})
    with urllib.request.urlopen(request,timeout=20) as response:
     if urllib.parse.urlsplit(response.geturl()).hostname!='i.redd.it':raise ValueError('Unsafe image redirect')
     if int(response.headers.get('Content-Length') or 0)>plan['perImageBytes']:raise ValueError('Original too large')
     content=response.read(plan['perImageBytes']+1)
    if not content or len(content)>plan['perImageBytes']:raise ValueError('Invalid original size')
    with Image.open(io.BytesIO(content)) as image:
     if image.format not in EXT or min(image.size)<128 or image.width*image.height>40_000_000 or getattr(image,'n_frames',1)>1:raise ValueError('Unsupported image geometry/format')
     image.verify();width,height=image.size;ext=EXT[image.format]
    if used+len(content)>plan['maxBytes']:raise ValueError('Total byte budget reached')
    data=content;break
   except Exception as error:errors.append({'url':url,'error':str(error)})
  if data is None:failures.append({'id':row['id'],'attempts':errors});continue
  sha=hashlib.sha256(data).hexdigest();relative=Path('objects')/f'{sha}.{ext}';target=OUT/relative
  if not target.exists():target.write_bytes(data)
  record={**row,'collectionUse':'development only','previewStoreRoot':str(ROOT/'datasets/reddit/data'),
   'original':{'path':str(relative),'sha256':sha,'bytes':len(data),'width':width,'height':height,'url':url},'originReview':'inherited source evidence; not independently verified'}
  with manifest.open('a') as stream:stream.write(json.dumps(record)+'\n')
  saved.append(record);seen.add(row['id']);counts[key]+=1;groups[row['sourceGroup']]+=1;used+=len(data)
  if len(saved)%10==0:print(json.dumps({'pairs':len(saved),'categories':dict(counts),'bytes':used}),flush=True)
  if all(counts[k]>=v for k,v in QUOTAS.items()):break
 summary={'pairs':len(saved),'categories':dict(counts),'bytes':used,'minutes':(time.monotonic()-start)/60,'failures':failures,'plan':plan}
 (REPORT/'pair-backfill.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='failures'}),flush=True)
if __name__=='__main__':main()
