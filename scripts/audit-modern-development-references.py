"""Hash/provenance checks for the separately prepared modern development data."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'datasets/modern-reference-development-2026-10-14/data';OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
def main():
 rows=[json.loads(x) for x in (BASE/'manifest.jsonl').read_text().splitlines()];existing=json.loads((ROOT/'backend/data/manifest.json').read_text());allowed={str((ROOT/'backend/data'/r['local']).resolve()) for r in existing if r['split']=='train'}
 protected={r['sha256'] for r in existing if r['split']!='train'};reserved_groups={'mllm:'+r['sourceGroup'] for r in existing if r['split']!='train'}
 old=json.loads((ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json').read_text())['rows'];protected.update(r['sourceHash'] for r in old if r['experimentSplit']!='development')
 frozen=json.loads((ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json').read_text())['rows'];protected.update((r.get(k) or {}).get('sha256') for r in frozen if r['split']!='development' for k in ['preview','original'] if (r.get(k) or {}).get('sha256'))
 for spec in json.loads((ROOT/'artifacts/external-benchmarks-2026-10-11/registry.json').read_text()).values():protected.update(r['sourceHash'] for r in json.loads(Path(spec['manifest']).read_text())['rows'])
 hash_labels=collections.defaultdict(set)
 for r in rows:hash_labels[r['sourceHash']].add(r['label'])
 findings=[];eligible=[]
 for r in rows:
  assert r['split']=='development' and r['sourceGroup'] not in reserved_groups
  path=Path(r['path'])
  if r['newDownload']:
   if path.is_symlink() or not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Unsafe new object')
  elif str(path.resolve()) not in allowed:raise ValueError('Existing reference outside train subset')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sourceHash']:raise ValueError('Changed reference')
  reasons=[]
  if r['sourceHash'] in protected:reasons.append('exact hash overlaps a reserved/evaluation asset')
  if len(hash_labels[r['sourceHash']])>1:reasons.append('identical bytes have conflicting publisher labels')
  if reasons:findings.append({'id':r['id'],'reasons':reasons})
  else:eligible.append(r)
 (BASE/'eligible-development.json').write_text(json.dumps({'note':'Development-only data for a future explicitly versioned training experiment; not used in the first student pilot. Publisher origin claims still need audit.','rows':eligible},indent=2))
 report={'references':len(rows),'hashesVerified':len(rows),'eligibleForFutureDevelopment':len(eligible),'reservedGroupOverlaps':0,'findings':findings,'generators':dict(collections.Counter(r['generator'] for r in eligible)),'note':'Reserved pixels were not opened; exclusions use existing metadata hashes. Source IDs are conservatively grouped across styles and generators. This is not an independent evaluation dataset.'}
 (OUT/'modern-reference-integrity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
