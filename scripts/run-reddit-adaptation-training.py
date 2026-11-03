"""Wait for GPU benchmarks, then run fixed fusion and small-student experiments."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
expected=set(json.loads((OUT/'external-config.json').read_text()));started=time.monotonic()
while True:
 history=OUT/'mcp-runs.json';runs=json.loads(history.read_text()) if history.exists() else [];completed=set()
 for row in runs:
  state=json.loads((Path(row['artifacts'])/'status.json').read_text())
  if state['status']=='complete':completed.add(row['dataset'])
  elif state['status']=='failed':raise RuntimeError(state)
 if completed>=expected:break
 if time.monotonic()-started>7200:raise RuntimeError('Benchmark wait exceeded two hours')
 print('Waiting for GPU benchmark:',len(completed),'of',len(expected),flush=True);time.sleep(30)
for module in ['research.reddit_adaptation.fusion','research.reddit_adaptation.student']:
 print('RUN',module,flush=True);subprocess.run([str(ROOT/'backend/.venv/bin/python'),'-m',module],cwd=ROOT,check=True,env={**os.environ,'LENS_DEVICE':'mps'})
