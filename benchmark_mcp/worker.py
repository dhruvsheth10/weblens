"""One isolated benchmark process using the existing model environment."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from .core import digest, implementation_hashes, read_json, write_json

def metrics(rows, cutoff):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels=np.array([r['label']=='ai' for r in rows])
    scores=np.array([r['score'] for r in rows])
    flags=scores>=cutoff
    tp,fp,tn,fn=[int(value.sum()) for value in (labels&flags,~labels&flags,~labels&~flags,labels&~flags)]
    categories={}
    for category in sorted({r['label']+'/'+r['group'] for r in rows}):
        values=[r for r in rows if r['label']+'/'+r['group']==category]
        flagged=sum(r['score']>=cutoff for r in values)
        categories[category]={'n':len(values),'flagged':flagged,'flagRate':flagged/len(values)}
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'accuracy':(tp+tn)/len(rows),'fpr':fp/(fp+tn) if fp+tn else 0,
            'auroc':float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
            'averagePrecision':float(average_precision_score(labels,scores)) if labels.any() else None,
            'categories':categories}

def run(directory):
    import fcntl
    import numpy as np
    request=read_json(directory/'request.json')
    project=Path(__file__).resolve().parents[1]
    if implementation_hashes(project,request['model']['model'])!=request['implementationHashes']:
        raise RuntimeError('Detector implementation changed after this run was requested.')
    status={'runId':request['runId'],'status':'starting','processed':0,'total':len(request['rows'])}
    write_json(directory/'status.json',status)
    # Separate Codex sessions can have separate MCP processes. Serialize model
    # workers across them so benchmark timing is not distorted by another run.
    gpu_lock=(directory.parent/'worker.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    started=time.perf_counter()
    model=request['model']
    if model['kind']=='profile' and model['profile']['kind'] in {'reddit-fusion','reddit-student'}:
        from backend.reddit_research import RedditFusionEngine,RedditStudentEngine
        engine=(RedditFusionEngine if model['profile']['kind']=='reddit-fusion' else RedditStudentEngine)(profile=model['profile'])
        cutoff=model['profile']['threshold']
    elif model['kind']=='profile' and model['profile']['kind'] in {'mirror','mirror-hybrid'}:
        from backend.mirror import MirrorEngine, MirrorCommunityEngine
        engine=(MirrorCommunityEngine if model['profile']['kind']=='mirror-hybrid' else MirrorEngine)(profile=model['profile'])
        cutoff=model['profile']['threshold']
    elif model['kind']=='profile' and model['profile']['kind'] in {'semantic-head','semantic-hybrid'}:
        from backend.community import CommunityEngine, CommunityFerretEngine
        engine=(CommunityFerretEngine if model['profile']['kind']=='semantic-hybrid' else CommunityEngine)(profile=model['profile'])
        cutoff=model['profile']['threshold']
    elif model['kind']=='profile':
        from backend.ferret_adapted import FerretNetAdaptedEngine
        engine=FerretNetAdaptedEngine(profile=model['profile'])
        cutoff=model['profile']['threshold']
    elif model['model']=='ferretnet-b-v1':
        from backend.ferretnet import FerretNetEngine, THRESHOLD
        engine=FerretNetEngine()
        cutoff=THRESHOLD
    elif model['model']=='ferretnet-opensynthid-v1':
        from backend.hybrid import HybridEngine, THRESHOLD
        engine=HybridEngine()
        cutoff=THRESHOLD
    else:
        from backend.engine import ImageEngine
        engine=ImageEngine()
        cutoff=engine.profile['threshold']
    startup_ms=(time.perf_counter()-started)*1000
    if request['thresholdOverride'] is not None:
        cutoff=request['thresholdOverride']
    results,failures=[],[]
    status.update(status='running',device=engine.device.type,startedAt=time.time())
    write_json(directory/'status.json',status)
    scan_start=time.perf_counter()
    root=Path(request['calibrationDirectory']).resolve()
    with (directory/'scores.jsonl').open('w') as stream:
        for index,row in enumerate(request['rows'],1):
            try:
                path=root/row['file']
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Image path escaped calibration.')
                data=path.read_bytes()
                if hashlib.sha256(data).hexdigest()!=row['sourceHash']:
                    raise ValueError('Image changed since benchmark request.')
                t=time.perf_counter()
                from backend.previews import predict_image
                prediction=predict_image(engine,data,request['variant'])
                elapsed=(time.perf_counter()-t)*1000
                if prediction['modelHash']!=model['modelHash']:
                    raise ValueError('Loaded model checksum differs from requested version.')
                result={**row,**prediction,'timingMs':elapsed,'flagged':prediction['score']>=cutoff}
                results.append(result)
                stream.write(json.dumps(result,allow_nan=False)+'\n')
                stream.flush()
            except Exception as error:
                failures.append({**row,'error':str(error)})
                write_json(directory/'failures.json',failures)
            status.update(processed=index,successful=len(results),failed=len(failures))
            write_json(directory/'status.json',status)
    if not results:
        raise RuntimeError('Every image failed. See failures.json; no accuracy claims are available.')
    elapsed=time.perf_counter()-scan_start
    if implementation_hashes(project,model['model'])!=request['implementationHashes']:
        raise RuntimeError('Detector implementation changed during this run; comparison refused.')
    times=np.array([r['timingMs'] for r in results])
    warm=times[1:] if len(times)>1 else times
    import torch
    import PIL
    from importlib.metadata import version
    runtime={'python':sys.version.split()[0],'torch':torch.__version__,'pillow':PIL.__version__,
             'platform':platform.platform(),'torchThreads':torch.get_num_threads(),
             'numpy':version('numpy'),'opencv':version('opencv-python-headless'),'pywavelets':version('PyWavelets'),'timm':version('timm')}
    summary={'schemaVersion':1,'runId':request['runId'],'model':model['model'],'modelHash':model['modelHash'],
             'implementationHashes':request['implementationHashes'],'implementationHash':digest(request['implementationHashes']),
             'pipeline':results[0]['pipeline'],'partition':request['partition'],'variant':request['variant'],
             'externalDataset':request.get('externalDataset'),
             'datasetHash':request['datasetHash'],'scoredFilesHash':digest(sorted(r['sourceHash']+r['file'] for r in results)),
             'device':engine.device.type,'runtime':runtime,'threshold':cutoff,'attempted':len(request['rows']),
             'successful':len(results),'failed':len(failures),'coverage':len(results)/len(request['rows']),
             'metrics':metrics(results,cutoff),'timing':{'startupMs':startup_ms,'firstImageMs':float(times[0]),
             'meanMs':float(times.mean()),'medianMs':float(np.median(times)),'p95Ms':float(np.quantile(times,.95)),
             'warmMeanMs':float(warm.mean()),'scanSeconds':elapsed},
             'warnings':['External dataset evaluation: thresholds were supplied, not fitted by this runner.' if request.get('externalDataset') else 'Calibration results support development, not independent validation.',
                         'Metrics exclude decode/inference failures; consult coverage and failures.']}
    write_json(directory/'summary.json',summary)
    write_json(directory/'failures.json',failures)
    fields=['file','label','group','sourceHash','score','logit','flagged','timingMs','model','modelHash','pipeline','width','height']
    for filename,selected in [('all-results.csv',results),('false-positives.csv',[r for r in results if r['label']=='real' and r['flagged']]),
                              ('false-negatives.csv',[r for r in results if r['label']=='ai' and not r['flagged']])]:
        with (directory/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(selected)
    status.update(status='complete',finishedAt=time.time())
    write_json(directory/'status.json',status)
    gpu_lock.close()

if __name__=='__main__':
    directory=Path(sys.argv[1])
    try:
        run(directory)
    except Exception as error:
        import traceback
        traceback.print_exc()
        status=read_json(directory/'status.json')
        status.update(status='failed',error=str(error),finishedAt=time.time())
        write_json(directory/'status.json',status)
        raise SystemExit(1)
