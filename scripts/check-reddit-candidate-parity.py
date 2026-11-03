"""Compare saved fitting predictions against independent MCP inference outputs."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.reddit_adaptation.common import OUT,load,write
from research.reddit_adaptation.student import inputs
from research.reddit_adaptation.fusion import assemble

def main():
 runs={r['candidate']:r for r in load(OUT/'candidate-mcp-runs.json') if r['dataset']=='reddit-tuning-preview'}
 expected={}
 p=load(OUT/'student-profile.json')
 expected['student']={r['sourceHash']:r['logit']-p['logitOffset'] for r in load(OUT/'student-tuning-predictions.json')['rows']}
 rows=inputs();features=np.load(OUT/'student-features.npz',allow_pickle=False)['features'];p=load(OUT/'student-head-profile.json')
 expected['student-head']={r['sourceHash']:float(features[i,0]@np.asarray(p['head']['weight'])+p['head']['bias']) for i,r in enumerate(rows) if r['domain']=='reddit' and r['split']=='tuning'}
 d=assemble()['reddit-tuning-preview'];p=load(OUT/'fusion-profile.json');logits=np.clip(d['x'],-p['clip'],p['clip'])[:,p['selectedFeatures']]@np.asarray(p['weight'])+p['bias']
 expected['fusion']={r['sourceHash']:float(v) for r,v in zip(d['rows'],logits)}
 results=[]
 for name,lookup in expected.items():
  run=runs[name];path=Path(run['artifacts'])
  if load(path/'status.json')['status']!='complete':raise ValueError('MCP candidate run is incomplete')
  actual=[json.loads(line) for line in (path/'scores.jsonl').read_text().splitlines()]
  if len(actual)!=783 or {r['sourceHash'] for r in actual}!=set(lookup):raise ValueError('Candidate coverage differs')
  differences=[abs(r['logit']-lookup[r['sourceHash']]) for r in actual]
  changed=sum((r['logit']>=0)!=(lookup[r['sourceHash']]>=0) for r in actual)
  result={'candidate':name,'runId':run['runId'],'n':len(actual),'maxAbsoluteLogitDifference':max(differences),'decisionDifferences':changed,'tolerance':1e-4}
  if changed or max(differences)>result['tolerance']:raise ValueError(result)
  results.append(result)
 write(OUT/'numerical-parity.json',{'passed':True,'comparisons':results,'note':'Saved training/analysis predictions versus fresh local MCP inference on identical authorized tuning bytes. Not Chrome parity.'})
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
