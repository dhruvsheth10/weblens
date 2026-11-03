"""Verify trained candidates with real MCP workers after all training has finished."""
import asyncio,json,os,subprocess,sys,time
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
async def main():
 started=time.monotonic()
 while True:
  status=OUT/'student-status.json'
  if status.exists() and json.loads(status.read_text()).get('status')=='complete':break
  if time.monotonic()-started>10800:raise RuntimeError('Training did not finish within three hours')
  await asyncio.sleep(30)
 subprocess.run([str(ROOT/'backend/.venv/bin/python'),str(ROOT/'scripts/enable-reddit-research-benchmarks.py')],cwd=ROOT,check=True)
 registry=json.loads((OUT/'external-config.json').read_text());registry.update(json.loads((ROOT/'artifacts/external-benchmarks-2026-10-11/registry.json').read_text()));config=OUT/'candidate-external-config.json';config.write_text(json.dumps(registry,indent=2))
 args=['-m','benchmark_mcp.server','--calibration','/Users/dhruvsheth/Downloads/benchmark/calibration','--manifest',str(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json'),'--manifest-sha','f6c0b73a767dd3fee4e44f87fb1d40e850daaf4608234cc45bd87806bf952d69','--device','mps','--external-config',str(config)]
 history=OUT/'candidate-mcp-runs.json';runs=json.loads(history.read_text()) if history.exists() else []
 async with stdio_client(StdioServerParameters(command=sys.executable,args=args,cwd=str(ROOT),env=dict(os.environ))) as (read,write):
  async with ClientSession(read,write) as session:
   await session.initialize()
   async def call(name,args):
    r=await session.call_tool(name,args)
    if r.is_error:raise RuntimeError(str(r.content))
    return r.structured_content or json.loads(r.content[0].text)
   print('SCOPE',json.dumps(await call('benchmark_info',{})),flush=True)
   profiles={}
   for name in ['fusion','student-head','student']:
    profiles[name]=await call('register_model',{'profile_path':str(OUT/f'{name}-profile.json')})
   jobs=[(name,'reddit-tuning-preview','original') for name in profiles]+[('student',name,variant) for name in ['park','synthetic-real','sensifake'] for variant in ['original','reddit-webp640']]
   for name,dataset,variant in jobs:
    prior=next((r for r in reversed(runs) if r['candidate']==name and r['dataset']==dataset and r.get('variant','original')==variant and json.loads((Path(r['artifacts'])/'status.json').read_text())['status']=='complete'),None)
    if prior:continue
    run=await call('start_benchmark',{'version':profiles[name]['version'],'dataset':dataset,'partition':'tuning','variant':variant});run.update(candidate=name,dataset=dataset,variant=variant);runs.append(run);history.write_text(json.dumps(runs,indent=2));print('START',json.dumps(run),flush=True)
    while True:
     result=await call('get_run',{'run_id':run['runId']});print(json.dumps({'candidate':name,'dataset':dataset,**{k:v for k,v in result.items() if k not in ['summary','failures']}}),flush=True)
     if result['status'] in ['complete','failed','cancelled']:break
     await asyncio.sleep(30)
    if result['status']!='complete':raise RuntimeError(result)
asyncio.run(main())
