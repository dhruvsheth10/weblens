"""Compile completed research measurements, including failures and deployment limits."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.reddit_adaptation.common import ROOT,OUT,load,write

def pct(x):return '—' if x is None else f'{x:.1%}'
def main():
 fusion=load(OUT/'fusion-results.json');student=load(OUT/'student-results.json');head=load(OUT/'student-head-results.json');training=load(OUT/'student-training.json')
 statuses=[];candidate_runs=[]
 for name,target in [('mcp-runs.json',statuses),('candidate-mcp-runs.json',candidate_runs)]:
  for r in load(OUT/name):
   status=load(Path(r['artifacts'])/'status.json')
   if status['status']!='complete':raise ValueError('A required run is incomplete')
   target.append({**r,'summary':load(Path(r['artifacts'])/'summary.json')})
 candidate_lookup={(r['candidate'],r['dataset']):r for r in candidate_runs}
 if not all((n,'reddit-tuning-preview') in candidate_lookup for n in ['fusion','student-head','student']):raise ValueError('Candidate MCP verification missing')
 student_mcp=candidate_lookup[('student','reddit-tuning-preview')]['summary'];head_mcp=candidate_lookup[('student-head','reddit-tuning-preview')]['summary']
 for key,manual,actual in [('student',student['categoryFpr01'],student_mcp['metrics']),('head',head['categoryFpr01'],head_mcp['metrics'])]:
  for metric in ['tp','fp','tn','fn']:
   if manual[metric]!=actual[metric]:raise ValueError(f'{key} numerical decision disagreement: {metric}')
 rows=[]
 for r in fusion['results']:
  if r['dataset']=='reddit-tuning-preview' and r['policy']=='tuning-category-fpr01':
   if any(x['name']==r['system'] for x in rows):continue
   rows.append({'name':r['system'],'metrics':r['metrics']})
 rows.extend([{'name':'New frozen-backbone head','metrics':head['categoryFpr01']},{'name':'Last-two-block supervised student','metrics':student['categoryFpr01']}])
 gate=[]
 for row in rows:
  missing=[c for c,m in row['metrics']['categories'].items() if m['realN'] and (m['fprWilson95'][1]>.01)]
  gate.append({'candidate':row['name'],'passed':False,'reasons':['Source-derived labels are not independent verified origin labels.','AI-meme and AI-screenshot evaluation categories are absent.']+(['95% image-level FPR upper bound exceeds 1% for: '+', '.join(missing)] if missing else [])})
 write(OUT/'deployment-gate.json',{'gate':'Public confident automatic badges','results':gate,'note':'The experiment remains useful for private research. A tuning-selected cutoff does not establish held-out reliability.'})
 lines=['# Reddit adaptation research — 2026-10-14','','The experiment evaluated the full frozen Reddit development/tuning data, fitted source-group-aware fusion, and trained two small-model research baselines locally. No reserved Reddit holdout, calibration audit or user validation pixels were evaluated. No paid API generations or rented GPU jobs were used.','',
 '## Primary comparison: actual Reddit tuning previews','','All rows below use an operating point selected on the same 783 tuning images to keep empirical false-positive rate at or below 1% in each observed real category. These are threshold-selection results, not independent validation. Labels remain source-derived.','',
 '| System | Precision | Recall | F1 | False flags | FPR | Precision at assumed 5% AI |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  m=r['metrics'];lines.append(f"| {r['name']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['fp']} | {pct(m['fpr'])} | {pct(m['precisionAtAssumedPrevalence']['0.05'])} |")
 lines+=['','## Frozen original operating points','','| System | Precision | Recall | F1 | False flags |','|---|---:|---:|---:|---:|']
 for r in fusion['results']:
  if r['dataset']=='reddit-tuning-preview' and r['policy']=='frozen0.5':
   m=r['metrics'];lines.append(f"| {r['system']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['fp']} |")
 for name,value in [('New head, raw zero logit',head['rawCut0']),('Fine-tuned student, raw zero logit',student['rawCut0'])]:lines.append(f"| {name} | {pct(value['precision'])} | {pct(value['recall'])} | {pct(value['f1'])} | {value['fp']} |")
 lines+=['','## Hard cases and real categories','','| System | AI hard cases detected | Real hard cases flagged | Photo FPR | Art FPR | Meme FPR | Screenshot FPR |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  cats=r['metrics']['categories'];h=cats.get('hard-case',{});lines.append(f"| {r['name']} | {h.get('tp',0)}/{h.get('aiN',0)} | {h.get('fp',0)}/{h.get('realN',0)} | "+' | '.join(pct(cats.get(c,{}).get('fpr')) for c in ['photo','art','meme','screenshot'])+' |')
 lines+=['','## Baseline benchmark coverage','','| Dataset | Attempted | Scored | Failed | Median ms |','|---|---:|---:|---:|---:|']
 for r in statuses:
  s=r['summary'];lines.append(f"| {r['dataset']} | {s['attempted']} | {s['successful']} | {s['failed']} | {s['timing']['medianMs']:.0f} |")
 lines+=['','Failures are retained in each run directory; unsupported originals do not count as successful negative predictions.','','## Existing external datasets: fine-tuned student','','The operating point is frozen from Reddit tuning. These datasets were not used to fit the new student or tune its threshold. Exact training-hash overlap checks are in external-overlap-audit.json. Prior project experiments have used these external datasets; they are not pristine final validation.','',
 '| Dataset | Input | N scored | Failures | Precision | Recall | F1 | FPR | Median ms |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in candidate_runs:
  if r['dataset']=='reddit-tuning-preview':continue
  s=r['summary'];m=s['metrics'];lines.append(f"| {r['dataset']} | {s['variant']} | {s['successful']} | {s['failed']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {pct(m['fpr'])} | {s['timing']['medianMs']:.0f} |")
 transfer=load(OUT/'student-calibration-transfer.json')
 lines+=['','## Calibration tuning transfer','','No threshold was fitted on these transfer images.','', '| Input | Candidate | Precision | Recall | F1 | False flags |','|---|---|---:|---:|---:|---:|']
 for r in transfer['results']:
  for key in ['head','student']:
   m=r[key];lines.append(f"| {r['variant']} | {key} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {m['fp']} |")
 audit=load(OUT/'label-audit-summary.json');pairs=load(OUT/'pair-backfill.json');integrity=load(OUT/'pair-integrity.json')
 lines+=['','## Data work completed','',f"- Evidence audit covered {audit['rows']} development/tuning images; {audit['excludedFromWeakTraining']} ambiguous development gallery panels were excluded from fitting.",f"- Review queue: {sum(audit['reviewQueue'].values())} entries, comprising previous disagreements and stratified random agreements. These are queued for origin review, not silently relabelled.",f"- Recovered {pairs['pairs']} development originals, {pairs['bytes']/1024/1024:.1f} MiB total: 40 each for real photos/art/memes/screenshots and AI art. All {integrity['hashChecks']} hash checks passed; {integrity['alignmentReviewRequired']} pairs failed the approximate alignment screen.",'- The new originals are in a separate manifest. They were not silently inserted into the frozen baseline or the predeclared student pilot.','- Source labels are weak evidence. Reviewing visual plausibility cannot establish origin; creator evidence and generator records remain needed.']
 modern=load(OUT/'modern-reference-collection.json');modern_integrity=load(OUT/'modern-reference-integrity.json')
 lines.append(f"- Prepared {modern['references']} public development references, including {modern['newImages']} newly downloaded images; {modern_integrity['hashesVerified']} hashes verified and {len(modern_integrity['findings'])} known reserved/evaluation collisions or contradictory duplicate labels. Generators: {modern['generators']}.")
 lines.append('- The modern supplement is a separate, future-training dataset. The first pilot was already frozen; these images did not enter its gradients or tuning. Source: https://huggingface.co/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset .')
 supplement=ROOT/'datasets/reddit/modern-development-2026-10-14/summary.json'
 if supplement.exists():
  s=load(supplement);lines.append(f"- New-post collection saved {s['images']} images. See collection.log for Reddit endpoint failures/backoff and the bounded collection outcome.")
 lines+=['','## Training and latency','',f"The pilot trained {training['trainableParameters']:,} parameters of {training['totalParameters']:,}, using {training['trainN']:,} development images and two fixed epochs of 4,096 sampled views. Backbone/head learning rates were 1e-5/1e-4. Sampling balanced dataset/class and capped rare-source weights. JPEG/WebP/resize/crop augmentation was applied to both labels. There were no teacher pseudo-labels or tuning gradients.",f"Training time: {training['seconds']/60:.1f} minutes. Fixed-checkpoint history: {training['history']}.",f"Fine-tuned student median full inference on actual Reddit previews: {student_mcp['timing']['medianMs']:.1f} ms; p95 {student_mcp['timing']['p95Ms']:.1f} ms. Cold worker startup: {student_mcp['timing']['startupMs']/1000:.1f} s.",f"Frozen-head candidate median inference: {head_mcp['timing']['medianMs']:.1f} ms.",'- No browser/WebGPU/Core ML latency claim is made from these Python measurements.','', '## Deployment gate and next decision','','No candidate passes the public confident-badge gate: origin labels remain weak, key AI content categories are absent, and the small real-category samples do not establish a ≤1% population false-positive rate. Wilson intervals and assumed-prevalence precision are in the JSON reports; correlated source groups make simple image-level intervals optimistic.','', 'Use the comparative results to choose the next research candidate. Review the queued disagreements plus agreements, add creator-proven modern photoreal and AI text/meme/screenshot examples, and use the paired originals for a separately versioned compression/distillation experiment. Preserve the Reddit holdout for one evaluation after model/threshold choices are fixed.','', '## Trying the experimental student','','The image lab and rebuilt extension expose Reddit Preview Student · experimental in Settings. Selecting it recommends displayed previews. The checkpoint is verified and loaded lazily by the local Python backend. Community + Ferret remains the default. Reload extension-dist before testing the new option. Scores are detector signals, not calibrated origin probabilities.','','## Reproduction and artifacts','','See docs/REDDIT-ADAPTATION.md. Raw scores and implementation/model hashes are retained in the MCP directories listed in mcp-runs.json and candidate-mcp-runs.json. Numerical parity is recorded in numerical-parity.json. The original production profiles and default are preserved.','']
 (OUT/'report.md').write_text('\n'.join(lines))
 write(OUT/'summary.json',{'primary':rows,'baselineRuns':[{k:v for k,v in r.items() if k!='summary'} for r in statuses],'candidateRuns':[{k:v for k,v in r.items() if k!='summary'} for r in candidate_runs],'deploymentGatePassed':False})
 print('\n'.join(lines[:17]))
if __name__=='__main__':main()
