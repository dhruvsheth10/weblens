"""Audit provenance without relabeling from detector predictions; reserve holdout."""
import collections,csv,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reddit-adaptation-2026-10-14'
spec=importlib.util.spec_from_file_location('reddit_labels',ROOT/'datasets/reddit/labels.py');labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)
def main():
 all_rows=json.loads((ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-splits.json').read_text())['rows']
 rows=[r for r in all_rows if r['split'] in {'development','tuning'}]
 post_sizes=collections.Counter(r['postId'] for r in all_rows)
 records=[]
 for r in rows:
  reasons=labels.needs_review(r)
  if r['basis'] in {'realorai-verified','isthisai-solved'} and post_sizes[r['postId']]>1:
   reasons.append('hard-case gallery needs panel-level provenance')
  records.append({'id':r['id'],'split':r['split'],'label':r['label'],'category':r['category'],'sourceGroup':r['sourceGroup'],
    'eligibleForWeakSupervision':not reasons,'reviewReasons':sorted(set(reasons)),
    'basis':r['basis'],'basisNote':r.get('basisNote',''),'postUrl':r['permalink'],'previewPath':str(ROOT/'datasets/reddit/data'/r['preview']['path']),
    'groundTruthStatus':'source-derived; not independently verified'})
 (OUT/'label-audit.json').write_text(json.dumps({'note':'Evidence audit only. Existing tuning labels are retained for primary comparisons. No reserved holdout pixels were opened.','rows':records},indent=2))
 previous=[]
 with (ROOT/'artifacts/mirror-evaluation-2026-10-13/reddit-review-queue.csv').open() as stream:previous=list(csv.DictReader(stream))
 lookup={r['id']:r for r in records};queue=[];selected=set()
 for r in previous:
  if r['id'] in lookup:queue.append({**lookup[r['id']],'selection':'previous MIRROR disagreement'});selected.add(r['id'])
 # Agreement sampling is independent of score confidence, stratified by class/category,
 # one member per source group and excludes the disagreement queue.
 previous_scores={}
 with (ROOT/'artifacts/mirror-evaluation-2026-10-13/comparison-scores.csv').open() as stream:
  for r in csv.DictReader(stream):
   if r['dataset']=='reddit-mirror' and r['system']=='MIRROR':previous_scores[Path(r['file']).stem]=float(r['score'])
 for label,category in sorted({(r['label'],r['category']) for r in records}):
  candidates=[r for r in records if r['id'] in previous_scores and r['id'] not in selected and r['label']==label and r['category']==category and ((previous_scores[r['id']]>=.5)==(label=='ai'))]
  candidates.sort(key=lambda r:hashlib.sha256(('agreement-review-20261014'+r['id']).encode()).hexdigest());groups=set()
  for r in candidates:
   if r['sourceGroup'] in groups:continue
   groups.add(r['sourceGroup']);selected.add(r['id']);queue.append({**r,'selection':'stratified random MIRROR agreement'})
   if len(groups)>=8:break
 with (OUT/'provenance-review.csv').open('w',newline='') as stream:
  fields=['id','split','label','category','selection','eligibleForWeakSupervision','reviewReasons','basis','basisNote','postUrl','previewPath','groundTruthStatus']
  writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(queue)
 summary={'rows':len(records),'excludedFromWeakTraining':sum(r['split']=='development' and not r['eligibleForWeakSupervision'] for r in records),
  'reviewRequired':dict(collections.Counter(r['split'] for r in records if r['reviewReasons'])),
  'reasons':dict(collections.Counter(reason for r in records for reason in r['reviewReasons'])),
  'reviewQueue':dict(collections.Counter(r['selection'] for r in queue)),
  'categoryLabelGaps':['No confirmed AI-meme or AI-screenshot categories in the frozen collection.','AI hard cases in tuning: only 22.','Source-based real labels still require creator evidence for a final deployment gate.']}
 (OUT/'label-audit-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
