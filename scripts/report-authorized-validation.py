"""Report frozen validation scores and metadata-only contamination checks."""
import argparse,csv,json,math,re,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark_mcp.core import read_json,write_json,digest
from benchmark_mcp.worker import metrics
OUT=ROOT/'artifacts/heldout-validation-2026-10-16'

def reviewed_audit():
 manifest=read_json(OUT/'manifest.json')
 if digest(manifest)!=read_json(OUT/'manifest-checksum.json')['sha256']:raise ValueError('Manifest checksum mismatch')
 rows=manifest['rows'];prior=read_json(ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json')['rows'];stems=defaultdict(set)
 for r in prior:stems[Path(r['file']).stem.lower()].add(r['experimentSplit'])
 hashes=defaultdict(list)
 for r in rows:hashes[r['sourceHash']].append(r)
 contradictions={h for h,rr in hashes.items() if len({r['label'] for r in rr})>1}
 exact_clean=[];strict_clean=[];seen=set();ambiguous=[];non_reddit_prefix=[]
 for r in rows:
  if r['knownPriorSourceIdOverlap'] and not re.fullmatch(r'[a-z0-9]{5,8}_\d{1,3}',Path(r['file']).stem):non_reddit_prefix.append(r['file'])
  if r['sourceHash'] in seen or r['sourceHash'] in contradictions or r['knownPriorExactOverlap']:continue
  seen.add(r['sourceHash']);exact_clean.append(r['file'])
  same_stem=sorted(stems.get(Path(r['file']).stem.lower(),[]))
  # A whole catalog/filename identity is useful; a shared prefix such as
  # artic_104221 and artic_104228 is not a Reddit source ID or same artwork.
  source_match=bool(r['knownPriorSourceIdOverlap']) and bool(re.fullmatch(r'[a-z0-9]{5,8}_\d{1,3}',Path(r['file']).stem))
  if same_stem or source_match:ambiguous.append({'file':r['file'],'sameStemPriorPartitions':same_stem,'recognizedSourceMatch':source_match})
  else:strict_clean.append(r['file'])
 def counts(files):
  selected=[r for r in rows if r['file'] in set(files)]
  return {'n':len(selected),'labels':dict(Counter(r['label'] for r in selected)),'categories':dict(Counter(r['label']+'/'+r['group'] for r in selected))}
 review={'raw':read_json(OUT/'data-audit.json'),'exactNewUnique':counts(exact_clean),'strictNewUnique':counts(strict_clean),'exactNewUniqueFiles':exact_clean,'strictNewUniqueFiles':strict_clean,'ambiguousNameMatches':ambiguous,'discardedBroadPrefixMatches':non_reddit_prefix,'note':'The initial broad source-prefix diagnostic overgrouped museum catalog filenames artic_<catalog ID>. These are distinct catalog items, not Reddit gallery frames. Exact overlaps remain unchanged. The stricter sensitivity subset also excludes ambiguous identical whole filenames. No score or image appearance informed this metadata correction. New means no detected prior exact bytes, not proven complete independence.'}
 path=OUT/'reviewed-data-audit.json'
 if path.exists() and read_json(path)!=review:raise ValueError('Frozen metadata-only audit changed')
 write_json(path,review)
 with (OUT/'known-overlap.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['file','label','group','sourceHash','priorPartitions']);writer.writeheader()
  for r in rows:
   if r['knownPriorExactOverlap']:writer.writerow({k:r[k] for k in ['file','label','group','sourceHash']}|{'priorPartitions':';'.join(r['knownPriorExactOverlap'])})
 return review

def wilson(k,n):
 if not n:return None
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,c-h),min(1,c+h)]

def extended(rows,threshold):
 m=metrics(rows,threshold);tp,fp,tn,fn=[m[k] for k in ['tp','fp','tn','fn']]
 if not tp+fp:m['precision']=None
 m.update(n=len(rows),specificity=tn/(tn+fp) if tn+fp else None,negativePredictiveValue=tn/(tn+fn) if tn+fn else None,falseNegativeRate=fn/(tp+fn) if tp+fn else None,precisionWilson95=wilson(tp,tp+fp),recallWilson95=wilson(tp,tp+fn),fprWilson95=wilson(fp,fp+tn))
 m['precisionAtAssumedPrevalence']={}
 for prevalence in [.01,.05,.1]:
  numerator=prevalence*m['recall'];denominator=numerator+(1-prevalence)*m['fpr'];m['precisionAtAssumedPrevalence'][str(prevalence)]=numerator/denominator if denominator else None
 return m

def pct(v):return '—' if v is None else f'{v*100:.2f}%'

def report(review):
 plan=read_json(OUT/'plan.json');runs=read_json(OUT/'runs.json');records=[]
 for variant in ['original','reddit-webp640','reddit-webp960']:
  for profile in ['community-ferret','reddit-student']:
   run=next(r for r in reversed(runs) if r['variant']==variant and r['profile']==profile and read_json(Path(r['artifacts'])/'status.json')['status']=='complete')
   directory=Path(run['artifacts']);summary=read_json(directory/'summary.json');request=read_json(directory/'request.json')
   if summary['modelHash']!=plan['models'][profile] or summary['threshold']!=plan['thresholds'][profile] or request['thresholdOverride'] is not None or summary['partition']!='validation':raise ValueError('Frozen model/cutoff/scope mismatch')
   scores=[json.loads(line) for line in (directory/'scores.jsonl').read_text().splitlines()]
   subsets={'full-folder':scores,'new-unique-bytes':[r for r in scores if r['file'] in set(review['exactNewUniqueFiles'])],'strict-new-unique':[r for r in scores if r['file'] in set(review['strictNewUniqueFiles'])]}
   for scope,selected in subsets.items():
    m=extended(selected,summary['threshold']);records.append({'scope':scope,**run,'metrics':m,'successful':len(selected),'attempted':summary['attempted'] if scope=='full-folder' else review['exactNewUnique' if scope=='new-unique-bytes' else 'strictNewUnique']['n'],'fullRunFailures':read_json(directory/'failures.json'),'timing':summary['timing'],'modelHash':summary['modelHash'],'threshold':summary['threshold'],'sourceScores':str(directory/'scores.jsonl')})
 for variant in ['original','reddit-webp640','reddit-webp960']:
  same=[r for r in records if r['scope']=='full-folder' and r['variant']==variant]
  summaries=[read_json(Path(r['artifacts'])/'summary.json') for r in same]
  if len({(s['datasetHash'],s['scoredFilesHash'],s['device'],digest(s['runtime'])) for s in summaries})!=1:raise ValueError('Unmatched model comparison files/device/runtime')
 write_json(OUT/'results.json',{'plan':plan,'reviewedAudit':{k:v for k,v in review.items() if k not in ['exactNewUniqueFiles','strictNewUniqueFiles','raw']},'records':records,'uncertaintyNote':'Wilson intervals assume independent images and correct labels. File deduplication does not remove all related content; real-source and generator coverage is uneven. Precision at a different prevalence assumes the measured error rates transfer.'})
 raw=review['raw'];lines=['# Explicitly authorized validation evaluation — 2026-10-16','','Both models and their 0.5 packaged operating points were frozen before scoring. No training, threshold fitting, cropping-policy selection or deployment promotion was performed. The routine MCP validation access remains disabled; a dedicated runner reused its immutable profiles, unchanged inference worker, progress reader and comparisons.','',f"The source folder contains {raw['files']:,} supported image files ({raw['labels']['ai']:,} AI / {raw['labels']['real']:,} real), {raw['uniqueHashes']:,} unique hashes. Exact overlap with earlier calibration experiments affects {sum(bool(r['exact']) for r in raw['priorOverlapDetails']):,} entries. Therefore the full folder is not an independent holdout.",'',f"The new unique byte subset contains {review['exactNewUnique']['n']} images: {review['exactNewUnique']['labels']}. Recall evidence is limited by the very small new AI sample. A stricter sensitivity subset removes ambiguous whole-filename matches and contains {review['strictNewUnique']['n']} images.",'','Unsupported files are excluded explicitly. Decode/inference failures are retained and excluded from classification denominators; neither failures nor unscored videos count as real predictions.','']
 for scope in ['full-folder','new-unique-bytes','strict-new-unique']:
  lines+=['## '+scope,'','| Model | Input | Scored / attempted | Precision | Recall | F1 | Accuracy | FPR | TP | FP | TN | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for r in records:
   if r['scope']!=scope:continue
   m=r['metrics'];lines.append(f"| {r['profile']} | {r['variant']} | {r['successful']} / {r['attempted']} | "+' | '.join(pct(m[k]) for k in ['precision','recall','f1','accuracy','fpr'])+' | '+' | '.join(str(m[k]) for k in ['tp','fp','tn','fn'])+' |')
  lines.append('')
 for scope in ['full-folder','new-unique-bytes']:
  lines+=['## '+scope+' categories','','The rate below is recall for AI rows and false-positive rate for real rows.','', '| Category | Input | Default flagged / N (rate) | Student flagged / N (rate) |','|---|---|---:|---:|']
  for variant in ['original','reddit-webp640','reddit-webp960']:
   pair={r['profile']:r for r in records if r['scope']==scope and r['variant']==variant}
   cats=set().union(*(r['metrics']['categories'] for r in pair.values()))
   for category in sorted(cats):
    cells=[]
    for name in ['community-ferret','reddit-student']:
     c=pair[name]['metrics']['categories'].get(category);cells.append(f"{c['flagged']} / {c['n']} ({pct(c['flagRate'])})" if c else '—')
    lines.append(f"| {category} | {variant} | "+' | '.join(cells)+' |')
  lines.append('')
 lines+=['## Inference timings','','Python/MPS timings include decode/inference and the selected WebP simulation, excluding source file reading/hashing and network/browser work. These are not Chrome inference timings.','', '| Model | Input | Cold load s | Median ms | p95 ms |','|---|---|---:|---:|---:|']
 for r in records:
  if r['scope']!='full-folder':continue
  t=r['timing'];lines.append(f"| {r['profile']} | {r['variant']} | {t['startupMs']/1000:.2f} | {t['medianMs']:.2f} | {t['p95Ms']:.2f} |")
 lines+=['','## Interpretation limits','','- Full-folder metrics mix previously used and new files; use the new unique subset for the less contaminated comparison.','- Only 25 AI files have new unique bytes before decode exclusions. A high or low recall estimate from that sample is uncertain.','- Exact hashes and recognizable source IDs do not prove complete independence, and released-model pretraining exposure is unknown.','- Folder labels are user-provided. No model disagreement was used to relabel them.','- Museum catalog prefixes are not Reddit source IDs. The raw broad-prefix diagnostic overgrouped 100 different artic catalog items; this was corrected using filename structure without consulting predictions.','- Wilson intervals, assumed-prevalence estimates, complete category counts, failures, model/source identities and raw score paths are in results.json.','- The dataset is now unblinded. Further development against these results would require fresh independent evaluation data.','']
 (OUT/'report.md').write_text('\n'.join(lines));print('\n'.join(lines[:10]))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args();review=reviewed_audit()
 if args.audit_only:print(json.dumps({k:review[k] for k in ['exactNewUnique','strictNewUnique','ambiguousNameMatches']},indent=2))
 else:report(review)
