"""One explicitly human-authorized held-out run; normal MCP boundaries stay locked.

Reuse immutable MCP profile snapshots, the unchanged benchmark worker, get_run
and compare_runs. Do not expose validation through the routine start_benchmark.
"""
import argparse,asyncio,hashlib,json,os,re,subprocess,sys,time,uuid
from collections import Counter,defaultdict
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark_mcp.core import digest,read_json,write_json,validate_profile,implementation_hashes
OUT=ROOT/'artifacts/heldout-validation-2026-10-16'
SOURCE=Path('/Users/dhruvsheth/Downloads/benchmark/validation')
VARIANTS=['original','reddit-webp640','reddit-webp960']
PROFILES=['community-ferret','reddit-student']
AUTHORIZATION='can you run the same thing on the other version of the benchmakr now- the validation set?'

def checksum(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if SOURCE.is_symlink() or SOURCE.resolve()!=SOURCE or not SOURCE.is_dir():raise ValueError('Validation root is missing or symlinked')
 profiles={name:validate_profile(read_json(ROOT/f'backend/models/{name}-profile.json')) for name in PROFILES}
 plan={'schemaVersion':1,'source':str(SOURCE),'authorization':AUTHORIZATION,'authorizationDate':'2026-10-16','models':{name:p['modelHash'] for name,p in profiles.items()},'thresholds':{name:p['threshold'] for name,p in profiles.items()},'variants':VARIANTS,'primary':'All supported validation image files; originals are primary and transformed variants are robustness checks, not independent images.','selection':'No threshold, model, preprocessing or hyperparameter fitting on validation.','reporting':'Report full-set coverage and failures, exact duplicates, known prior-data overlap, clean unique subset, per-category and prevalence-adjusted metrics.','runnerSha256':checksum(Path(__file__))}
 if (OUT/'plan.json').exists() and read_json(OUT/'plan.json')!=plan:raise ValueError('The frozen validation plan changed')
 write_json(OUT/'plan.json',plan)
 for name,p in profiles.items():write_json(OUT/f'{name}-profile.json',p)
 if (OUT/'manifest.json').exists():
  manifest=read_json(OUT/'manifest.json')
  if digest(manifest)!=read_json(OUT/'manifest-checksum.json')['sha256']:raise ValueError('Validation manifest changed')
  return manifest
 rows=[];skipped=[];extensions={'.png','.jpg','.jpeg','.jfif','.webp','.avif'}
 for label in ['ai','real']:
  base=SOURCE/label
  if base.is_symlink() or not base.is_dir():raise ValueError('Validation needs ai/ and real/ directories')
  for directory,children,files in os.walk(base,followlinks=False):
   if any((Path(directory)/name).is_symlink() for name in children+files):raise ValueError('Symlinked validation entries are refused')
   for name in sorted(files):
    path=Path(directory)/name;relative=path.relative_to(SOURCE)
    if path.suffix.lower() not in extensions:
     skipped.append({'file':str(relative),'reason':'Unsupported extension; no video inference'});continue
    rows.append({'file':str(relative),'label':label,'group':relative.parts[1] if len(relative.parts)>2 else 'unspecified','sourceHash':checksum(path),'bytes':path.stat().st_size})
 rows.sort(key=lambda r:r['file'])
 if not rows or {r['label'] for r in rows}!={'ai','real'}:raise ValueError('Both validation labels are required')
 known=defaultdict(set);known_posts=defaultdict(set)
 def record(h,file,scope):
  if h:known[h].add(scope)
  match=re.fullmatch(r'([a-z0-9]{5,8})_\d+',Path(file).stem)
  if match:known_posts[match[1]].add(scope)
 for r in read_json(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json')['rows']:record(r['sourceHash'],r['file'],'calibration-'+r['experimentSplit'])
 for r in read_json(ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json')['rows']:
  if r['split'] not in {'development','tuning'}:continue
  for role in ['preview','original']:
   asset=r.get(role) or {}
   if asset.get('sha256'):record(asset['sha256'],r.get('id','')+'.jpg','reddit-'+r['split'])
 by_hash=defaultdict(list);overlaps=[]
 for r in rows:
  by_hash[r['sourceHash']].append(r)
  match=re.fullmatch(r'([a-z0-9]{5,8})_\d+',Path(r['file']).stem)
  exact=sorted(known.get(r['sourceHash'],[]));source=sorted(known_posts.get(match[1],[])) if match else []
  r['knownPriorExactOverlap']=exact;r['knownPriorSourceIdOverlap']=source
  if exact or source:overlaps.append({'file':r['file'],'exact':exact,'sourceId':source})
 conflicts=[{'sourceHash':h,'files':[r['file'] for r in rr]} for h,rr in by_hash.items() if len({r['label'] for r in rr})>1]
 manifest={'schemaVersion':1,'source':str(SOURCE),'planHash':digest(plan),'rows':rows,'skippedFiles':skipped}
 write_json(OUT/'manifest.json',manifest);write_json(OUT/'manifest-checksum.json',{'sha256':digest(manifest)})
 write_json(OUT/'data-audit.json',{'files':len(rows),'uniqueHashes':len(by_hash),'labels':dict(Counter(r['label'] for r in rows)),'categories':dict(Counter(r['label']+'/'+r['group'] for r in rows)),'bytes':sum(r['bytes'] for r in rows),'withinValidationDuplicateEntries':len(rows)-len(by_hash),'contradictoryExactLabels':conflicts,'priorOverlapEntries':len(overlaps),'priorOverlapDetails':overlaps,'skippedFiles':skipped,'limitations':['Exact-byte and recognizable Reddit source-ID checks do not establish complete visual/source independence.','Pretraining exposure cannot be audited from the released checkpoints.','Folder labels are supplied by the user; visual plausibility is not used to relabel.']})
 return manifest

async def run(manifest):
 args=['-m','benchmark_mcp.server','--calibration','/Users/dhruvsheth/Downloads/benchmark/calibration','--manifest',str(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json'),'--manifest-sha','f6c0b73a767dd3fee4e44f87fb1d40e850daaf4608234cc45bd87806bf952d69','--device','mps']
 history=OUT/'runs.json';runs=read_json(history) if history.exists() else []
 async with stdio_client(StdioServerParameters(command=sys.executable,args=args,cwd=str(ROOT),env=dict(os.environ))) as (read,write):
  async with ClientSession(read,write) as session:
   await session.initialize()
   async def call(name,args):
    result=await session.call_tool(name,args)
    if result.is_error:raise RuntimeError(str(result.content))
    return result.structured_content or json.loads(result.content[0].text)
   info=await call('benchmark_info',{});write_json(OUT/'mcp-scope.json',info)
   if info['validationAccess'] is not False:raise ValueError('Routine MCP guard unexpectedly changed')
   for name in PROFILES:
    p=read_json(OUT/f'{name}-profile.json');registered=await call('register_model',{'profile_path':str(OUT/f'{name}-profile.json')})
    if registered['version']!=p['modelHash']:raise ValueError('Registered model identity mismatch')
   # Complete both original-image runs before the slower compression checks.
   for variant in VARIANTS:
    for name in PROFILES:
     prior=next((r for r in reversed(runs) if r['profile']==name and r['variant']==variant and read_json(Path(r['artifacts'])/'status.json')['status']=='complete'),None)
     if prior:continue
     if digest(read_json(OUT/'manifest.json'))!=read_json(OUT/'manifest-checksum.json')['sha256']:raise ValueError('Validation manifest changed')
     p=read_json(OUT/f'{name}-profile.json');model={'kind':'profile','model':p['model'],'modelHash':p['modelHash'],'profile':p}
     run_id=uuid.uuid4().hex;directory=ROOT/'artifacts/mcp-benchmarks'/run_id;directory.mkdir()
     rows=[{k:r[k] for k in ['file','label','group','sourceHash']} for r in manifest['rows']]
     request={'schemaVersion':1,'runId':run_id,'calibrationDirectory':str(SOURCE),'partition':'validation','externalDataset':'human-authorized-validation-2026-10-16','model':model,'thresholdOverride':None,'variant':variant,'rows':rows,'datasetHash':digest(rows),'requestedAt':time.time(),'implementationHashes':implementation_hashes(ROOT,p['model']),'validationAuthorization':{'quote':AUTHORIZATION,'planPath':str(OUT/'plan.json'),'planHash':manifest['planHash'],'manifestHash':read_json(OUT/'manifest-checksum.json')['sha256']}}
     write_json(directory/'request.json',request);write_json(directory/'status.json',{'runId':run_id,'status':'queued','total':len(rows),'processed':0})
     entry={'runId':run_id,'profile':name,'model':p['model'],'variant':variant,'artifacts':str(directory)};runs.append(entry);write_json(history,runs)
     env=dict(os.environ,LENS_DEVICE='mps',PYTHONUNBUFFERED='1')
     print('START',json.dumps(entry),flush=True)
     with (directory/'worker.log').open('w') as log:
      process=subprocess.Popen([str(ROOT/'backend/.venv/bin/python'),'-m','benchmark_mcp.worker',str(directory)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
      try:
       while True:
        status=await call('get_run',{'run_id':run_id});print(json.dumps({**entry,**{k:v for k,v in status.items() if k not in {'summary','failures'}}}),flush=True)
        if status['status'] in {'complete','failed','cancelled'}:break
        if process.poll() is not None:raise RuntimeError('Worker exited without final status; inspect '+str(directory))
        await asyncio.sleep(30)
       process.wait(timeout=10)
       if status['status']!='complete' or process.returncode:raise RuntimeError(status)
      except BaseException:
       if process.poll() is None:process.terminate();process.wait(timeout=30)
       raise
   comparisons={}
   for variant in VARIANTS:
    ids=[next(r['runId'] for r in reversed(runs) if r['profile']==name and r['variant']==variant and read_json(Path(r['artifacts'])/'status.json')['status']=='complete') for name in PROFILES]
    comparisons[variant]=await call('compare_runs',{'run_ids':ids})
   write_json(OUT/'mcp-comparisons.json',comparisons);print('COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 manifest=prepare();print(json.dumps(read_json(OUT/'data-audit.json')),flush=True)
 if not args.prepare_only:asyncio.run(run(manifest))
