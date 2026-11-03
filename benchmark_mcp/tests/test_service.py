from __future__ import annotations
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_mcp.core import BenchmarkManager, CHECKPOINT, digest, validate_profile, write_json, read_json

PROJECT=Path(__file__).resolve().parents[2]

def profile(**overrides):
    value={'schemaVersion':1,'checkpointSha256':CHECKPOINT,'kind':'fixed','representation':'native5_mean',
           'model':'ferretnet-test-v1','pipeline':'test-native5-v1','threshold':.5,**overrides}
    value['modelHash']=digest(value)
    return value

class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=PROJECT/'artifacts',prefix='mcp-unit-')
        self.root=Path(self.temp.name)
        self.calibration=self.root/'calibration'
        for label in ('real','ai'):
            (self.calibration/label/'group').mkdir(parents=True)
            (self.calibration/label/'group'/'sample.jpg').write_bytes(label.encode())
        self.manager=BenchmarkManager(PROJECT,self.calibration,self.root/'runs')
    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()
    def test_mirror_profiles_register_and_reject_policy_mutations(self):
        from backend.mirror_profile import profile as mirror_profile
        for blend in [False,True]:
            value=mirror_profile(blend)
            path=self.root/'mirror.json';write_json(path,value)
            registered=self.manager.register(path)
            self.assertEqual(registered['version'],value['modelHash'])
            value['threshold']=.15;value['modelHash']=digest({k:v for k,v in value.items() if k!='modelHash'})
            write_json(path,value)
            with self.assertRaises(ValueError):self.manager.register(path)

    def test_reddit_research_numeric_profile_is_registered_immutably(self):
        from backend.reddit_research_profile import digest as research_digest
        from backend.mirror_profile import profile as mirror_profile,COMMUNITY_SHA
        p={'schemaVersion':1,'kind':'reddit-fusion','model':'reddit-fusion-test-v1','pipeline':'clipped-component-logits-standardized-logistic-v1','components':['community_forensics','ferretnet_b','mirror'],'selectedFeatures':[0,2],'clip':20.,'weight':[.3,.7],'bias':-.2,'threshold':.5,'mirrorHash':mirror_profile()['modelHash'],'communityHash':COMMUNITY_SHA}
        p['modelHash']=research_digest(p);path=self.root/'reddit.json';write_json(path,p)
        registered=self.manager.register(path);self.assertEqual(registered['version'],p['modelHash'])
        p['components']=['subreddit','filename','mirror'];p['modelHash']=research_digest(p);write_json(path,p)
        with self.assertRaises(ValueError):self.manager.register(path)

    def test_only_class_directories_are_enumerated(self):
        original_walk=os.walk
        roots=[]
        def walk(path,**kwargs):
            roots.append(Path(path))
            return original_walk(path,**kwargs)
        with patch('benchmark_mcp.core.os.walk',walk):
            rows=self.manager.dataset('calibration')
        self.assertEqual(len(rows),2)
        self.assertEqual(roots,[self.calibration/'real',self.calibration/'ai'])
        self.assertFalse(self.manager.info()['validationAccess'])
    def test_validation_and_path_traversal_are_rejected_before_access(self):
        with patch('benchmark_mcp.core.os.walk',side_effect=AssertionError('Must not enumerate')):
            for partition in ('validation','audit','../validation'):
                with self.assertRaises(ValueError):
                    self.manager.dataset(partition)
        for path in ('../validation/x.jpg','/tmp/x.jpg','real/validation/x.jpg'):
            with self.assertRaises(ValueError):
                self.manager.safe_image(path)
        with self.assertRaises(ValueError):
            BenchmarkManager(PROJECT,self.root/'validation',self.root/'bad-output')
    def test_symlinks_are_rejected(self):
        (self.calibration/'real'/'linked.jpg').symlink_to(self.calibration/'ai'/'group'/'sample.jpg')
        with self.assertRaises(ValueError):
            self.manager.dataset('calibration')
    def test_manifest_locks_audit_and_checks_source_bytes(self):
        rows=self.manager.dataset('calibration')
        manifest=self.root/'manifest.json'
        write_json(manifest,{'rows':[dict(r,file='calibration/'+r['file'],experimentSplit='development' if i==0 else 'audit') for i,r in enumerate(rows)]})
        manager=BenchmarkManager(PROJECT,self.calibration,self.root/'manifest-runs',manifest)
        self.assertEqual(len(manager.dataset('research')),1)
        for partition in ('audit','validation','calibration'):
            with self.assertRaises(ValueError): manager.dataset(partition)
        self.manager.safe_image(rows[0]['file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            manager.dataset('research')

    def test_manifest_boundaries_are_checksummed(self):
        manifest=self.root/'manifest.json'
        write_json(manifest,{'rows':[]})
        with self.assertRaisesRegex(ValueError,'checksum'):
            BenchmarkManager(PROJECT,self.calibration,self.root/'wrong-manifest',manifest,manifest_sha='0'*64)
        manager=BenchmarkManager(PROJECT,self.calibration,self.root/'pinned-manifest',manifest)
        write_json(manifest,{'rows':[{'experimentSplit':'audit'}]})
        with self.assertRaisesRegex(ValueError,'changed'):
            manager.dataset('research')
    def test_external_registry_is_pinned_and_cannot_unlock_validation(self):
        manifest=self.root/'external-manifest.json'
        write_json(manifest,{'rows':[dict(r,file='calibration/'+r['file'],experimentSplit='tuning') for r in self.manager.dataset('calibration')]})
        config=self.root/'external.json'
        entry={'root':str(self.calibration),'manifest':str(manifest),'manifestSha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
        write_json(config,{'fixture':entry})
        manager=BenchmarkManager(PROJECT,self.calibration,self.root/'ext-runs',external_config=config)
        self.assertEqual(len(manager.external['fixture'].dataset('tuning')),2)
        with self.assertRaisesRegex(ValueError,'Unknown external'):
            manager.start('ferretnet-b-v1',dataset='unknown')
        with self.assertRaisesRegex(ValueError,'frozen partition'):
            manager.start('ferretnet-b-v1',dataset='fixture',partition='validation')
        write_json(config,{'fixture':dict(entry,root=str(self.root/'validation'/'calibration'))})
        with self.assertRaisesRegex(ValueError,'outside validation'):
            BenchmarkManager(PROJECT,self.calibration,self.root/'bad-ext',external_config=config)
        link=self.root/'linked';link.symlink_to(self.root,target_is_directory=True)
        write_json(config,{'fixture':dict(entry,root=str(link/'calibration'))})
        with self.assertRaisesRegex(ValueError,'Symlinked'):
            BenchmarkManager(PROJECT,self.calibration,self.root/'bad-ext-link',external_config=config)
    def test_profiles_are_immutable_and_numerical(self):
        path=self.root/'profile.json'
        value=profile()
        write_json(path,value)
        version=self.manager.register(str(path))['version']
        value=profile(threshold=.7)
        write_json(path,value)
        other=self.manager.register(str(path))['version']
        self.assertNotEqual(version,other)
        self.assertEqual(len(self.manager.models()),5)
        for invalid in (profile(kind='python'),{'threshold':float('nan')},
                        profile(kind='probe',head={'weight':[0]*12,'bias':0}),profile(model='ferretnet-b-v1')):
            with self.assertRaises((ValueError,TypeError)):
                validate_profile(invalid)
        tampered=profile()
        tampered['threshold']=.2
        with self.assertRaisesRegex(ValueError,'checksum'):
            validate_profile(tampered)
    def test_semantic_profile_is_checksummed_immutable_and_numerical(self):
        value=read_json(PROJECT/'backend/models/community-profile.json')
        path=self.root/'semantic.json';write_json(path,value)
        registered=self.manager.register(path)
        self.assertEqual(registered['version'],value['modelHash'])
        self.assertEqual(self.manager.models()[-1]['representation'],'community-cls384')
        value['head']['bias']+=1
        with self.assertRaises(ValueError):validate_profile(value)
        self.assertEqual(read_json(registered['snapshot'])['modelHash'],registered['version'])
    def test_failure_to_start_is_reported(self):
        self.manager.python=self.root/'missing-python'
        with self.assertRaises(OSError): self.manager.start('ferretnet-b-v1')
        self.assertEqual(self.manager.runs()[0]['status'],'failed')
    def test_bad_run_ids_and_thresholds_are_rejected(self):
        for run_id in ('../profile','validation','0'*64):
            with self.assertRaises(ValueError): self.manager.get(run_id)
        for cutoff in (-.1,1.1,float('nan'),True):
            with self.assertRaises(ValueError): self.manager.start('ferretnet-b-v1',threshold=cutoff)
        with self.assertRaises(ValueError): self.manager.compare([])

    def test_all_image_failures_are_reported_without_accuracy(self):
        # The unit fixtures deliberately contain invalid JPEG data. This runs
        # a real worker and confirms errors never turn into synthetic scores.
        started=self.manager.start('ferretnet-b-v1')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            status=self.manager.get(started['runId'])
            if status['status']=='failed': break
            time.sleep(.1)
        self.assertEqual(status['status'],'failed',str(status))
        self.assertEqual(status['failed'],2)
        self.assertNotIn('summary',status)
        self.assertEqual(len(json.loads((Path(status['artifacts'])/'failures.json').read_text())),2)

    def test_single_active_job_and_cancel(self):
        started=self.manager.start('ferretnet-b-v1')
        with self.assertRaisesRegex(ValueError,'already active'):
            self.manager.start('ferretnet-b-v1')
        cancelled=self.manager.cancel(started['runId'])
        self.assertEqual(cancelled['status'],'cancelled')
        self.assertEqual(self.manager.cancel(started['runId'])['status'],'cancelled')

class StdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_handshake_compatibility(self):
        from mcp import Client,StdioServerParameters
        with tempfile.TemporaryDirectory(dir=PROJECT/'artifacts',prefix='mcp-handshake-') as temporary:
            root=Path(temporary)
            (root/'calibration').mkdir()
            parameters=StdioServerParameters(command=str(PROJECT/'benchmark_mcp/.venv/bin/python'),
                args=[str(PROJECT/'benchmark_mcp/server.py'),'--calibration',str(root/'calibration'),'--output',str(root/'runs')])
            async with Client(parameters,mode='legacy',raise_exceptions=True) as client:
                listing=await client.list_tools()
                self.assertEqual(len(listing.tools),8)
                result=await client.call_tool('benchmark_info',{})
                self.assertFalse(result.is_error)
                self.assertFalse(result.structured_content['validationAccess'])

    async def test_real_stdio_protocol_and_real_models(self):
        from mcp import Client, StdioServerParameters
        # Minimal PNG fixture creation uses the ML environment so the MCP
        # environment stays independent of Pillow/Torch/sklearn.
        import subprocess
        with tempfile.TemporaryDirectory(dir=PROJECT/'artifacts',prefix='mcp-integration-') as temporary:
            root=Path(temporary)
            calibration=root/'calibration'
            subprocess.run([str(PROJECT/'backend/.venv/bin/python'),'-c',
                'from PIL import Image; from pathlib import Path; import sys; '
                'p=Path(sys.argv[1]); '
                '[(p/c).mkdir(parents=True) for c in ("real","ai")]; '
                '[Image.effect_noise((320,320),20+i*30).convert("RGB").save(p/c/(str(i)+".png")) for i,c in enumerate(("real","ai"))]',
                str(calibration)],check=True)
            # This is a synthetic trap, not the user's validation directory.
            (root/'validation').mkdir()
            trap=root/'validation'/'must-not-read.jpg'
            trap.write_bytes(b'untouched')
            trap.chmod(0)
            parameters=StdioServerParameters(command=str(PROJECT/'benchmark_mcp/.venv/bin/python'),
                args=[str(PROJECT/'benchmark_mcp/server.py'),'--calibration',str(calibration),'--output',str(root/'runs'),'--device','cpu'])
            async with Client(parameters,raise_exceptions=True) as client:
                listing=await client.list_tools()
                self.assertEqual(len(listing.tools),8)
                async def call(name,arguments=None):
                    result=await client.call_tool(name,arguments or {})
                    self.assertFalse(result.is_error,str(result))
                    return result.structured_content
                info=await call('benchmark_info')
                self.assertFalse(info['validationAccess'])
                invalid=await client.call_tool('start_benchmark',{'version':'ferretnet-b-v1','partition':'validation'})
                self.assertTrue(invalid.is_error)
                self.assertIn('validation',invalid.content[0].text.lower())
                async def finish(run_id):
                    for _ in range(240):
                        result=await call('get_run',{'run_id':run_id})
                        if result['status'] in ('complete','failed','cancelled'):
                            self.assertEqual(result['status'],'complete',str(result))
                            return result
                        await asyncio.sleep(.25)
                    self.fail('Worker timeout')
                first=await call('start_benchmark',{'version':'ferretnet-b-v1'})
                done=await finish(first['runId'])
                self.assertEqual(done['summary']['successful'],2)
                self.assertEqual(done['summary']['modelHash'],CHECKPOINT)
                hybrid=await call('start_benchmark',{'version':'ferretnet-opensynthid-v1'})
                hybrid_done=await finish(hybrid['runId'])
                self.assertEqual(hybrid_done['summary']['successful'],2)
                self.assertEqual(hybrid_done['summary']['threshold'],.5)
                semantic_path=root/'semantic.json'
                write_json(semantic_path,read_json(PROJECT/'backend/models/community-profile.json'))
                semantic_registration=await call('register_model',{'profile_path':str(semantic_path)})
                semantic=await call('start_benchmark',{'version':semantic_registration['version'],'variant':'reddit-webp640'})
                semantic_done=await finish(semantic['runId'])
                self.assertEqual(semantic_done['summary']['successful'],2)
                self.assertEqual(semantic_done['summary']['model'],'community-preview-v1')
                preview=await call('start_benchmark',{'version':'ferretnet-b-v1','variant':'reddit-webp640'})
                await finish(preview['runId'])
                preview_comparison=await call('compare_runs',{'run_ids':[semantic['runId'],preview['runId']]})
                self.assertEqual(len(preview_comparison['runs']),2)
                combination_path=root/'combination.json'
                write_json(combination_path,read_json(PROJECT/'backend/models/community-ferret-profile.json'))
                combination_registration=await call('register_model',{'profile_path':str(combination_path)})
                combination=await call('start_benchmark',{'version':combination_registration['version'],'variant':'reddit-webp640'})
                combination_done=await finish(combination['runId'])
                self.assertEqual(combination_done['summary']['successful'],2)
                self.assertEqual(combination_done['summary']['model'],'community-ferret-preview-v1')
                second=await call('start_benchmark',{'version':'ferretnet-b-v1','threshold':.25})
                await finish(second['runId'])
                comparison=await call('compare_runs',{'run_ids':[first['runId'],second['runId']]})
                self.assertEqual(len(comparison['runs']),2)
                self.assertEqual(comparison['runs'][1]['threshold'],.25)
                version_path=root/'profile.json'
                write_json(version_path,profile())
                registered=await call('register_model',{'profile_path':str(version_path)})
                third=await call('start_benchmark',{'version':registered['version']})
                custom=await finish(third['runId'])
                self.assertEqual(custom['summary']['modelHash'],registered['version'])
                self.assertEqual(custom['summary']['model'],'ferretnet-test-v1')
                self.assertEqual(len(custom['summary']['implementationHash']),64)
                fourth=await call('start_benchmark',{'version':'ferretnet-b-v1','variant':'jpeg75'})
                await finish(fourth['runId'])
                mismatch=await client.call_tool('compare_runs',{'run_ids':[first['runId'],fourth['runId']]})
                self.assertTrue(mismatch.is_error)
                cancelled=await call('start_benchmark',{'version':'ferretnet-b-v1'})
                self.assertEqual((await call('cancel_run',{'run_id':cancelled['runId']}))['status'],'cancelled')
                legacy=await call('start_benchmark',{'version':'fiser-dear-r-v1'})
                legacy_done=await finish(legacy['runId'])
                self.assertEqual(legacy_done['summary']['model'],'fiser-dear-r-v1')
                self.assertEqual(legacy_done['summary']['modelHash'],'81c73e3fcf7112629a384a22f088b98d9138edad127e4d267ff08daaa5c16331')
                history=await call('list_runs')
                self.assertEqual(len(history['runs']),10)
            evidence={'protocol':'real stdio discover/list/tools/call; initialize-handshake separately tested','tools':8,'realModelRuns':5,'cancelledRuns':1,
                      'customProfileRun':True,'validationRequestRejected':True,'comparisonsChecked':True}
            write_json(PROJECT/'artifacts/ferretnet-improvement-2026-10-11/mcp-integration-evidence.json',evidence)

if __name__=='__main__':
    unittest.main()
