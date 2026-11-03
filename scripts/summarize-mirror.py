"""Compare frozen component scores on the exact same completed MCP rows."""
import csv,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_mcp.worker import metrics
from backend.image_io import sigmoid
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/mirror-evaluation-2026-10-13'
manifest=json.loads((ROOT/'artifacts/ferretnet-improvement-2026-10-11/manifest.json').read_text())
partitions={str(Path(r['file']).relative_to('calibration')):r['experimentSplit'] for r in manifest['rows'] if r['experimentSplit'] in {'development','tuning'}}
ferret=.8304235935211183;offset=math.log(ferret/(1-ferret));uncertain=.03761442731955501
summaries=[];rowsout=[]
for run in json.loads((OUT/'mcp-runs.json').read_text()):
 directory=Path(run['artifacts']);summary=json.loads((directory/'summary.json').read_text());status=json.loads((directory/'status.json').read_text())
 if status['status']!='complete':raise RuntimeError('All runs must finish before analysis')
 rows=[json.loads(x) for x in (directory/'scores.jsonl').read_text().splitlines()]
 dataset=summary.get('externalDataset') or 'local'
 for split in (['tuning'] if dataset!='local' else ['development','tuning']):
  selected=rows if dataset!='local' else [r for r in rows if partitions[r['file']]==split]
  systems={name:[] for name in ['Community + Ferret','MIRROR','MIRROR + Community + Ferret','Uncertain-only MIRROR (simulation)']}
  escalations=0
  for r in selected:
   c=r['components'];base=max(c['community_forensics']['logit'],c['ferretnet_b']['logit']-offset);mirror=c['mirror']['logit']
   assert abs(r['logit']-max(base,mirror))<1e-6
   escalate=base<0 and c['community_forensics']['score']>=uncertain;escalations+=escalate
   logits=[base,mirror,max(base,mirror),max(base,mirror) if escalate else base]
   timing=r['componentTimingsMs'];times=[timing['community'],timing['mirror'],r['timingMs'],timing['community']+(timing['mirror'] if escalate else 0)]
   for (name,values),logit,t in zip(systems.items(),logits,times):
    value={**r,'score':sigmoid(logit),'logit':logit,'measuredOrEstimatedMs':t};values.append(value)
    rowsout.append({'dataset':dataset,'split':split,'variant':summary['variant'],'system':name,'file':r['file'],'label':r['label'],'group':r['group'],'sourceHash':r['sourceHash'],'score':value['score'],'logit':logit,'timingMs':t})
  for name,values in systems.items():
   times=sorted(r['measuredOrEstimatedMs'] for r in values)
   summaries.append({'dataset':dataset,'split':split,'variant':summary['variant'],'system':name,'n':len(values),'metrics':metrics(values,.5),'timing':{'medianMs':statistics.median(times),'p95Ms':times[min(len(times)-1,int(.95*len(times)))]},'mcpRunId':run['runId'],'startupMs':summary['timing']['startupMs'],'coverage':summary['coverage'],'failed':summary['failed'],'escalatedFraction':escalations/len(values) if name.startswith('Uncertain') else None})
(OUT/'comparison.json').write_text(json.dumps(summaries,indent=2))
with (OUT/'comparison-scores.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rowsout[0]));writer.writeheader();writer.writerows(rowsout)
lines=['# MIRROR evaluation — 2026-10-13','', 'All decisions use frozen 0.5 cuts. No threshold fitting or training was performed. The original Community/Ferret component thresholds are unchanged. Validation, the reserved calibration audit, and the new Reddit holdout were not scored.','', '## Results','', '| Dataset / split | Input | System | N | Precision | Recall | F1 | False flags | Median ms |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for s in summaries:
 m=s['metrics'];lines.append(f"| {s['dataset']} / {s['split']} | {s['variant']} | {s['system']} | {s['n']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} | {m['fp']} | {s['timing']['medianMs']:.0f} |")
lines+=['','## Per-category tuning results','','| Dataset | Input | System | Label/category | N | Flagged | Flag rate |','|---|---|---|---|---:|---:|---:|']
for s in summaries:
 if s['split']!='tuning' or s['system'].startswith('Uncertain'):continue
 for category,m in s['metrics']['categories'].items():lines.append(f"| {s['dataset']} | {s['variant']} | {s['system']} | {category} | {m['n']} | {m['flagged']} | {m['flagRate']:.1%} |")
lines+=['','## Interpretation limits','','- Development images fitted the current Community head, so development scores are descriptive training-set behavior, not evidence of generalization. Tuning has already informed earlier decisions. Neither is independent final validation.','- Reddit results use a fixed, category-balanced tuning sample with source-derived labels. It is not a random feed sample; its precision does not estimate real Reddit prevalence. Rules and self-reported answers can be wrong, and gallery panels can have mixed provenance.','- JPEG75 and WebP640 apply the same transformations to both classes. They simulate re-encoding. The Reddit sample uses actual downloaded preview bytes.','- Component accuracy is reconstructed from their exact saved logits in the combination run. Component latency is recorded within that run; the cascade latency is an additive estimate and excludes network downloads, queueing and first-load cost.','- The uncertain-only cascade is a frozen policy simulation: keep existing strong flags; run MIRROR only when the old combination is below 0.5 and the semantic score is at least 0.03761442731955501. It is not shipped as a new default.','- All input failures remain in the MCP failure manifests; metrics use only successfully scored images. Consult coverage in comparison.json.','', '## Reproduction','','See docs/MIRROR.md and scripts/run-mirror-mcp.py. Raw rows, failure lists, model/implementation hashes and timing metadata remain in each MCP run listed in mcp-runs.json. An early uncached run (136cca1bfa8b4fd49bb0cac5a6318a9d) was cancelled after 271 images and is excluded.','']
(OUT/'report.md').write_text('\n'.join(lines))
for s in summaries:
 if s['split']=='tuning':print(s['dataset'],s['variant'],s['system'],{k:round(s['metrics'][k],4) for k in ['precision','recall','f1','fpr']},s['metrics']['fp'],s['timing'])

# Keep review work separate from training labels. Predictions never relabel data.
source_rows={r['id']:r for r in json.loads((OUT/'reddit-splits.json').read_text())['rows']}
sample_rows={str(Path(r['file']).relative_to('calibration')):r for r in json.loads((OUT/'reddit/manifest.json').read_text())['rows']}
review=[]
for r in rowsout:
 if r['dataset']!='reddit-mirror' or r['system']!='MIRROR' or (r['score']>=.5)==(r['label']=='ai'):continue
 sample=sample_rows[r['file']];source=source_rows[sample['redditId']]
 review.append({'id':source['id'],'issue':'false-positive against source label' if r['label']=='real' else 'false-negative against source label',
     'sourceLabel':r['label'],'reviewedLabel':'','category':r['group'],'mirrorScore':r['score'],'labelBasis':source['basis'],
     'basisNote':source.get('basisNote',''),'subreddit':source['subreddit'],'postUrl':source['permalink'],
     'imagePath':str(OUT/'reddit/calibration'/r['file']),'reviewEvidence':''})
with (OUT/'reddit-review-queue.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
with (OUT/'report.md').open('a') as stream:
 stream.write('''
## Recommendation and shipped behavior

Keep Community + Ferret as the default and keep MIRROR as an optional local second
opinion. Do not promote the full OR blend to confident automatic badges yet.
On local original tuning images, MIRROR alone detects only 57/114 GPT examples (50%),
while the current combination detects 92/114 (80.7%); the full blend detects 101/114
(88.6%). They are complementary, rather than a simple standalone replacement.

The uncertainty-gated simulation is the most promising precision-preserving next
experiment: local original recall 82.6% with 5 false flags, and WebP640 recall 66.0%
with zero false flags on this tuning set. It is a saved-score policy simulation,
not an implemented new selectable detector. On the Reddit sample it reaches only
37.5% recall at 84.4% precision and requires MIRROR on 16.3% of images. The gate can
miss images that the small model confidently scores low. Network fetch policy is
a separate decision and these numbers do not include it.

Postpone distillation until the hard cases and label quality are improved. MIRROR
finds 40/50 ordinary AI-art examples in the Reddit sample but only 4/22 AI hard cases;
it also flags 4/10 source-labelled real hard cases. Blindly copying its predictions
would transfer those mistakes to the small student. `reddit-review-queue.csv`
contains 40 MIRROR disagreements with the source labels (12 flagged real, 28 missed
AI), with local paths, post links and existing evidence. The review labels are
intentionally empty: a human must establish origin, not copy either model's guess.
No production or training labels were changed by this evaluation.

After reviewing those disagreements and a random agreement sample, use the frozen
Reddit development split for supervised fine-tuning of the existing small student,
with identical compression distributions for both labels and balanced source/category
sampling. Add MIRROR soft targets only as an auxiliary loss on reviewed development
data; keep true labels authoritative. Select model/thresholds on tuning, then evaluate
the untouched Reddit holdout once. Preserve the user's separate validation folder
for the final agreed evaluation. No student training was performed in this task.

## Resources and verification

MIRROR has 855,964,610 parameters and a 3,453,063,720-byte local phase-2 checkpoint.
The combination used about 3.56 GB of allocated MPS tensor memory and 4.38 GB of MPS
driver allocation (the latter includes tensor allocation; do not add them). This is
viable as a local Mac backend, not a compact self-contained Chrome model. Standalone
load took 8.5–10.4 seconds in direct checks; full cold benchmark-worker startup took
14.9–16.5 seconds. On the actual Reddit previews, median warm component time was
215 ms for MIRROR and 53 ms for Community + Ferret; the full blend median was 268 ms.
Synthetic WebP timings also include image re-encoding and should not be mistaken
for the cost of scoring an already-downloaded Reddit preview.

The app now offers MIRROR H+ and the full MIRROR + Community + Ferret combination.
Both are experimental and load on demand. The default and old numerical profiles
remain unchanged. The standalone FerretNet-B and FiSeR + DEAR-r picker/API choices
are retired, with their implementations, weights, benchmark identities and historical
recording access preserved. The Chrome package still uses the local Python backend.

Preprocessing parity, CPU/MPS parity, cached/uncached forward parity, API/recording
contracts, browser interaction evidence, and the test/build logs are alongside this
report. See `implementation-checks.md`. 6,075 image/variant evaluations completed,
with 100% coverage and zero failures. Repeated variants are not independent images.

Sources: [MIRROR reference model](https://github.com/handsome-rich/MIRROR/blob/adb928ec56c4ffd77c53ccb908bea29bb559922c/models/mirror.py),
[authors' evaluation preprocessing](https://github.com/handsome-rich/MIRROR/blob/adb928ec56c4ffd77c53ccb908bea29bb559922c/data/datasets.py),
[DINOv3 H+ architecture](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py).
''')
