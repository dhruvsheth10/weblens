"""Dependency-free benchmark coordinator. No model code executes in the MCP process."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

CHECKPOINT = 'fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2'
DIMENSIONS = {'center256':384,'native5_mean':384,'native9_mean':384,'native5_mean_std':768,
              'multiscale_native':768,'resize5_mean':384,'native_resize_means':768,'all_scales':1152}
FIXED = {'center256','center512','native_resize_blend'} | {
    f'{prefix}_{method}' for prefix in ('native5','native9','resize5') for method in ('mean','median','max')}
BUILTINS = {'ferretnet-b-v1':{'kind':'builtin','model':'ferretnet-b-v1','modelHash':CHECKPOINT},
            'ferretnet-opensynthid-v1':{'kind':'builtin','model':'ferretnet-opensynthid-v1',
                                      'modelHash':'6fafa513baa3f68af32dc9809251f138ec084b6ab486adf5d0a2495689eb3389'},
            'fiser-dear-r-v1':{'kind':'builtin','model':'fiser-dear-r-v1',
                              'modelHash':'81c73e3fcf7112629a384a22f088b98d9138edad127e4d267ff08daaa5c16331'}}
EXTENSIONS = {'.png','.jpg','.jpeg','.webp','.avif','.bmp','.tif','.tiff','.gif'}

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def implementation_hashes(project, model):
    files=['backend/assets.py','backend/previews.py','backend/image_io.py','benchmark_mcp/core.py','benchmark_mcp/worker.py']
    if model.startswith('reddit-'):
        files+=['backend/reddit_research.py','backend/reddit_research_profile.py','backend/mirror.py','backend/mirror_profile.py','backend/vendor/mirror/upstream.py','backend/community.py','backend/semantic_profile.py','backend/ferretnet.py','backend/models/community-profile.json','backend/models/community-ferret-profile.json']
    elif model.startswith('mirror-'):
        files+=['backend/mirror.py','backend/mirror_profile.py','backend/vendor/mirror/upstream.py','backend/community.py','backend/semantic_profile.py','backend/ferretnet.py','backend/models/community-profile.json','backend/models/community-ferret-profile.json']
    elif model=='fiser-dear-r-v1':
        files+=['backend/engine.py']
    elif model.startswith('community-'):
        files+=['backend/community.py','backend/semantic_profile.py','backend/ferretnet.py','backend/models/community-profile.json','backend/models/community-ferret-profile.json']
    else:
        files+=['backend/image_io.py','backend/ferretnet.py','backend/ferret_adapted.py']
        if model=='ferretnet-opensynthid-v1':
            files+=['backend/hybrid.py','backend/vendor/opensynthid.py','backend/models/hybrid-profile.json']
    return {name:hashlib.sha256((Path(project)/name).read_bytes()).hexdigest() for name in files}

def validate_profile(profile):
    if not isinstance(profile,dict):
        raise ValueError('Profile must be a JSON object.')
    if profile.get('kind') in {'reddit-fusion','reddit-student'}:
        from backend.reddit_research_profile import validate_profile as validate_reddit
        return validate_reddit(profile)
    if profile.get('kind') in {'mirror','mirror-hybrid'}:
        from backend.mirror_profile import validate_profile as validate_mirror
        return validate_mirror(profile)
    if profile.get('kind') in {'semantic-head','semantic-hybrid'}:
        from backend.semantic_profile import validate_profile as validate_semantic
        return validate_semantic(profile)
    if profile.get('schemaVersion') != 1 or profile.get('checkpointSha256') != CHECKPOINT:
        raise ValueError('Only profiles for the pinned FerretNet-B checkpoint are supported.')
    if profile.get('kind') not in ('probe','fixed'):
        raise ValueError('Profile must be a numerical probe or fixed aggregation; executable models are not accepted.')
    name = profile.get('representation')
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    if profile['kind'] == 'probe':
        head = profile.get('head',{})
        if not isinstance(head,dict):
            raise ValueError('Head must be a numerical JSON object.')
        weights = head.get('weight')
        if name not in DIMENSIONS or not isinstance(weights,list) or len(weights)!=DIMENSIONS[name] or not all(map(finite,weights)) or not finite(head.get('bias')):
            raise ValueError('Invalid numerical head dimensions or non-finite parameters.')
    elif name not in FIXED:
        raise ValueError('Unknown crop aggregation.')
    if not finite(profile.get('threshold')) or not 0 <= profile['threshold'] <= 1:
        raise ValueError('Threshold must be in [0,1].')
    if not isinstance(profile.get('model'),str) or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,79}',profile['model']) or profile['model'] in BUILTINS:
        raise ValueError('Use a unique versioned model name, not a builtin model identity.')
    if not isinstance(profile.get('pipeline'),str) or not profile['pipeline']:
        raise ValueError('Pipeline identity is required.')
    if profile.get('modelHash') != digest({k:v for k,v in profile.items() if k!='modelHash'}):
        raise ValueError('Profile checksum mismatch. Recompute after editing any parameter.')
    return profile

class BenchmarkManager:
    def __init__(self, project, calibration, output, manifest=None, python=None, device=None, manifest_sha=None, external_config=None):
        self.project = Path(project).resolve()
        candidate = Path(calibration).absolute()
        if candidate.is_symlink() or candidate.name != 'calibration' or 'validation' in [p.lower() for p in candidate.parts]:
            raise ValueError('Dataset must be an explicit calibration directory. Validation is disabled.')
        self.calibration = candidate.resolve(strict=True)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'profiles').mkdir(exist_ok=True)
        self.manifest = Path(manifest).resolve(strict=True) if manifest else None
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest() if self.manifest else None
        if manifest_sha and self.manifest_sha != manifest_sha:
            raise ValueError('Split manifest checksum mismatch. Refusing to change reserved audit boundaries.')
        self.python = Path(python or self.project/'backend/.venv/bin/python')
        self.device = device
        self.processes = {}
        self.lock = threading.RLock()
        self.external = {}
        self.external_name = None
        if external_config:
            config_path = Path(external_config).resolve(strict=True)
            if not config_path.is_relative_to(self.project):
                raise ValueError('External dataset registry must be inside the project.')
            for name,spec in read_json(config_path).items():
                if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',name) or name == 'local':
                    raise ValueError('Invalid external dataset name.')
                root = Path(spec['root']).absolute()
                manifest_path = Path(spec['manifest']).absolute()
                for path in (root,manifest_path):
                    if not path.is_relative_to(self.project) or '..' in path.parts or 'validation' in [p.lower() for p in path.parts]:
                        raise ValueError('External datasets must have explicit project-local paths, outside validation.')
                    for ancestor in [path,*path.parents]:
                        if ancestor == self.project.parent:
                            break
                        if ancestor.is_symlink():
                            raise ValueError('Symlinked external dataset paths are refused.')
                    if not path.resolve(strict=True).is_relative_to(self.project):
                        raise ValueError('External dataset escaped the project.')
                self.external[name] = BenchmarkManager(self.project,root,self.output,manifest_path,
                    python=self.python,device=self.device,manifest_sha=spec['manifestSha256'])
                self.external[name].external_name = name

    def info(self):
        return {'calibrationDirectory':str(self.calibration),'validationAccess':False,
                'splitManifestSha256':self.manifest_sha,
                'auditAccess':False if self.manifest else 'No audit split configured',
                'partitions':['research','development','tuning'] if self.manifest else ['calibration'],
                'outputDirectory':str(self.output),'profileFormat':'Pinned FerretNet-B, Community Forensics, MIRROR, or Reddit research profiles',
                'variants':['original','jpeg75','resize512','reddit-webp640','reddit-webp960'],
                'externalDatasets':{name:{'root':str(m.calibration),'manifestSha256':m.manifest_sha,
                                         'partitions':['tuning']} for name,m in self.external.items()},
                'instructions':'Start returns immediately. Poll get_run. Benchmark scores are not calibrated probabilities. Never tune from audit or validation.'}

    def models(self):
        result = list(BUILTINS.values())
        for path in sorted((self.output/'profiles').glob('*.json')):
            profile = validate_profile(read_json(path))
            result.append({'kind':profile['kind'],'model':profile['model'],'modelHash':profile['modelHash'],
                           'version':profile['modelHash'],'representation':profile.get('representation','component-logits' if profile['kind']=='reddit-fusion' else 'community-vit-s224-finetuned' if profile['kind']=='reddit-student' else 'mirror-h-1280' if profile['kind'].startswith('mirror') else 'community-cls384'),'threshold':profile['threshold']})
        return result

    def register(self, profile_path):
        source = Path(profile_path).absolute()
        if not source.is_relative_to(self.project) or source.suffix != '.json' or source.is_symlink() or not source.resolve().is_relative_to(self.project):
            raise ValueError('Profile must be a regular JSON file inside this project.')
        if source.stat().st_size > 1_000_000:
            raise ValueError('Profile exceeds the numerical-head size limit.')
        profile = validate_profile(read_json(source))
        destination = self.output/'profiles'/f'{profile["modelHash"]}.json'
        with self.lock:
            if not destination.exists():
                write_json(destination,profile)
        return {'version':profile['modelHash'],'model':profile['model'],'snapshot':str(destination),'threshold':profile['threshold']}

    def safe_image(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts or 'validation' in [p.lower() for p in relative.parts]:
            raise ValueError('Calibration paths cannot escape their root or refer to validation.')
        path = self.calibration/relative
        for parent in [path,*path.parents]:
            if parent == self.calibration.parent:
                break
            if parent.is_symlink():
                raise ValueError('Symlinked dataset entries are refused.')
        if not path.resolve(strict=True).is_relative_to(self.calibration) or not path.is_file():
            raise ValueError('Invalid calibration image path.')
        return path

    def dataset(self, partition):
        rows = []
        if self.manifest:
            if hashlib.sha256(self.manifest.read_bytes()).hexdigest()!=self.manifest_sha:
                raise ValueError('Split manifest changed after server startup.')
            if partition not in {'research','development','tuning'}:
                raise ValueError('Only development, tuning or research are enabled; audit and validation are locked.')
            for row in read_json(self.manifest)['rows']:
                if row['experimentSplit'] not in ({'development','tuning'} if partition=='research' else {partition}):
                    continue
                relative = Path(row['file'])
                if relative.parts[0] != 'calibration':
                    raise ValueError('Split manifest contains a non-calibration image.')
                rows.append({'file':str(relative.relative_to('calibration')),'label':row['label'],'group':row['group'],
                             'sourceHash':row['sourceHash']})
        else:
            if partition != 'calibration':
                raise ValueError('Validation/audit access is disabled. Use the configured calibration partition.')
            # Enumerate only these two class directories, never the dataset parent.
            for label in ('real','ai'):
                base = self.calibration/label
                if base.is_symlink() or not base.is_dir():
                    raise ValueError('Calibration requires real/ and ai/ directories.')
                for directory, children, files in os.walk(base,followlinks=False):
                    if any((Path(directory)/name).is_symlink() for name in children):
                        raise ValueError('Symlinked dataset directories are refused.')
                    for name in sorted(files):
                        path=Path(directory)/name
                        if path.suffix.lower() in EXTENSIONS:
                            relative=path.relative_to(self.calibration)
                            rows.append({'file':str(relative),'label':label,'group':relative.parts[1] if len(relative.parts)>2 else 'unspecified'})
        for row in rows:
            if row['label'] not in ('real','ai'):
                raise ValueError('Labels must be real or ai.')
            path=self.safe_image(row['file'])
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            if 'sourceHash' in row and checksum!=row['sourceHash']:
                raise ValueError('Image changed since the fixed calibration manifest: '+row['file'])
            row['sourceHash']=checksum
        if not rows:
            raise ValueError('No calibration images in this partition.')
        return sorted(rows,key=lambda r:r['file'])

    def _run_directory(self, run_id):
        if not re.fullmatch(r'[0-9a-f]{32}',run_id):
            raise ValueError('Invalid run ID.')
        path=self.output/run_id
        if not path.is_dir():
            raise ValueError('Unknown run ID.')
        return path

    def get(self, run_id):
        with self.lock:
            directory=self._run_directory(run_id)
            status=read_json(directory/'status.json')
            process=self.processes.get(run_id)
            if process and process.poll() is not None and status['status'] in {'queued','starting','running'}:
                status.update(status='failed',error=f'Worker exited with code {process.returncode}; see worker.log',finishedAt=time.time())
                write_json(directory/'status.json',status)
            if (directory/'summary.json').exists():
                status['summary']=read_json(directory/'summary.json')
            status['artifacts']=str(directory)
            return status

    def runs(self, limit=20):
        if not 1 <= limit <= 100:
            raise ValueError('Limit must be between 1 and 100.')
        directories=sorted((p for p in self.output.iterdir() if re.fullmatch(r'[0-9a-f]{32}',p.name) and p.is_dir()),
                           key=lambda p:p.stat().st_mtime,reverse=True)
        return [self.get(p.name) for p in directories[:limit]]

    def start(self, version, partition=None, threshold=None, variant='original', dataset='local'):
        with self.lock:
            for run_id,process in self.processes.items():
                if process.poll() is None:
                    status=self.get(run_id)
                    if status['status'] in {'complete','failed','cancelled'}:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            raise ValueError('Previous worker is still shutting down: '+run_id)
                    else:
                        raise ValueError('A benchmark is already active: '+run_id)
            if dataset != 'local':
                if dataset not in self.external:
                    raise ValueError('Unknown external dataset; register it explicitly in the server configuration.')
                if partition not in (None,'tuning'):
                    raise ValueError('External evaluation uses its frozen partition; no threshold fitting is performed.')
                run = self.external[dataset].start(version,'tuning',threshold,variant)
                self.processes[run['runId']] = self.external[dataset].processes[run['runId']]
                return run
            partition=partition or ('research' if self.manifest else 'calibration')
            if variant not in ('original','jpeg75','resize512','reddit-webp640','reddit-webp960'):
                raise ValueError('Unknown image variant.')
            if threshold is not None and (isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1):
                raise ValueError('Threshold must be finite and between 0 and 1.')
            if version in BUILTINS:
                model=BUILTINS[version]
            elif re.fullmatch(r'[a-f0-9]{64}',version or ''):
                profile_path=self.output/'profiles'/f'{version}.json'
                profile=validate_profile(read_json(profile_path))
                model={'kind':'profile','model':profile['model'],'modelHash':profile['modelHash'],'profile':profile}
            else:
                raise ValueError('Unknown model version. Register a profile first or use a builtin ID.')
            rows=self.dataset(partition)
            run_id=uuid.uuid4().hex
            directory=self.output/run_id
            directory.mkdir()
            request={'schemaVersion':1,'runId':run_id,'calibrationDirectory':str(self.calibration),'partition':partition,
                     'externalDataset':self.external_name,
                     'model':model,'thresholdOverride':threshold,'variant':variant,'rows':rows,
                     'datasetHash':digest(rows),'requestedAt':time.time()}
            request['implementationHashes']=implementation_hashes(self.project,model['model'])
            write_json(directory/'request.json',request)
            write_json(directory/'status.json',{'runId':run_id,'status':'queued','total':len(rows),'processed':0})
            environment=os.environ.copy()
            environment['PYTHONUNBUFFERED']='1'
            if self.device:
                environment['LENS_DEVICE']=self.device
            try:
                with (directory/'worker.log').open('w') as log:
                    process=subprocess.Popen([str(self.python),'-m','benchmark_mcp.worker',str(directory)],cwd=self.project,
                                             env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                self.processes[run_id]=process
            except OSError as error:
                write_json(directory/'status.json',{'runId':run_id,'status':'failed','error':str(error),'processed':0,'total':len(rows)})
                raise
            return self.get(run_id)

    def cancel(self, run_id):
        with self.lock:
            status=self.get(run_id)
            if status['status'] in {'complete','failed','cancelled'}:
                return status
            process=self.processes.get(run_id)
            if process is None:
                raise ValueError('This server does not own that worker.')
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            directory=self._run_directory(run_id)
            status=read_json(directory/'status.json')
            status.update(status='cancelled',finishedAt=time.time())
            write_json(directory/'status.json',status)
            return self.get(run_id)

    def compare(self, run_ids):
        if not 2 <= len(run_ids) <= 10 or len(set(run_ids)) != len(run_ids):
            raise ValueError('Compare two to ten distinct complete runs.')
        runs=[self.get(run_id) for run_id in run_ids]
        if any(run['status']!='complete' for run in runs):
            raise ValueError('Only complete runs can be compared.')
        summaries=[run['summary'] for run in runs]
        identities={(s['datasetHash'],s['partition'],s['variant'],s['device'],s['scoredFilesHash']) for s in summaries}
        if len(identities)!=1:
            raise ValueError('Runs must use the same image hashes, partition, variant, device, and successfully scored files.')
        if len({digest(s.get('runtime',{})) for s in summaries})!=1:
            raise ValueError('Runs use different runtime/library environments; re-run them before comparing latency.')
        baseline=summaries[0]
        return {'baselineRun':run_ids[0],'runs':[{
            'runId':run_id,'model':summary['model'],'modelHash':summary['modelHash'],'threshold':summary['threshold'],
            'implementationHash':summary['implementationHash'],'runtime':summary.get('runtime'),
            'metrics':summary['metrics'],'timing':summary['timing'],
            'delta':{key:summary['metrics'][key]-baseline['metrics'][key] for key in ('precision','recall','f1','fpr')},
        } for run_id,summary in zip(run_ids,summaries)],
            'note':'Thresholds are reported explicitly; timings include decode and inference, exclude model startup. No automatic model or threshold selection.'}

    def close(self):
        for run_id,process in list(self.processes.items()):
            if process.poll() is None:
                self.cancel(run_id)
