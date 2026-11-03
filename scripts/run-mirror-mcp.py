"""Use a fresh stdio MCP instance so new profile adapters need no desktop restart."""
import asyncio,json,os,sys
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/mirror-evaluation-2026-10-13'
async def main():
 args=['-m','benchmark_mcp.server','--calibration','/Users/dhruvsheth/Downloads/benchmark/calibration','--manifest',str(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json'),'--manifest-sha','f6c0b73a767dd3fee4e44f87fb1d40e850daaf4608234cc45bd87806bf952d69','--device','mps','--external-config',str(OUT/'external-config.json')]
 async with stdio_client(StdioServerParameters(command=sys.executable,args=args,cwd=str(ROOT),env=dict(os.environ))) as (read,write):
  async with ClientSession(read,write) as session:
   await session.initialize()
   async def call(name,args):
    r=await session.call_tool(name,args)
    if r.is_error:raise RuntimeError(str(r.content))
    return r.structured_content or json.loads(r.content[0].text)
   info=await call('benchmark_info',{});(OUT/'mcp-info.json').write_text(json.dumps(info,indent=2))
   model=await call('register_model',{'profile_path':str(ROOT/'backend/models/mirror-community-profile.json')})
   (OUT/'mcp-registration.json').write_text(json.dumps(model,indent=2))
   runs=[]
   for dataset,partition,variant in [('local','research',v) for v in ['original','jpeg75','reddit-webp640']]+[('reddit-mirror','tuning','original')]:
    run=await call('start_benchmark',{'version':model['version'],'dataset':dataset,'partition':partition,'variant':variant});runs.append(run)
    (OUT/'mcp-runs.json').write_text(json.dumps(runs,indent=2));print('START',json.dumps(run),flush=True)
    while True:
     result=await call('get_run',{'run_id':run['runId']})
     print(json.dumps({'variant':variant,'dataset':dataset,**{k:v for k,v in result.items() if k not in ['summary','request','failures']}}),flush=True)
     if result['status'] in ['complete','failed','cancelled']:break
     await asyncio.sleep(30)
    if result['status']!='complete':raise RuntimeError(result)
asyncio.run(main())
