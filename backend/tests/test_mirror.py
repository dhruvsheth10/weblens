import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image,ImageFile
from fastapi.testclient import TestClient
from backend.mirror import preprocess,MirrorCommunityEngine
from backend.mirror_profile import profile,validate_profile
from backend.image_io import sigmoid

class MirrorContractTests(unittest.TestCase):
    def test_preprocessing_matches_pinned_upstream_on_geometry_and_png_path(self):
        path=Path(__file__).resolve().parents[2]/'artifacts/mirror-evaluation-2026-10-13/upstream-dataset.py'
        spec=importlib.util.spec_from_file_location('mirror_dataset_reference',path);ref=importlib.util.module_from_spec(spec)
        old=ImageFile.LOAD_TRUNCATED_IMAGES
        try:spec.loader.exec_module(ref)
        finally:ImageFile.LOAD_TRUNCATED_IMAGES=old
        pre=ref.Get_Transforms(None,'preprocess')[1];norm=ref.Get_Transforms(None,'normal')[1]
        rng=np.random.default_rng(2026)
        for size in [(128,199),(224,224),(320,512),(513,777),(1200,701),(1400,1400)]:
            image=Image.fromarray(rng.integers(0,256,(size[1],size[0],3),dtype=np.uint8))
            for png in [False,True]:
                expected=pre(image)
                if png:expected=ref.compress_image(expected,96)
                torch.testing.assert_close(preprocess(image,png),norm(expected),rtol=0,atol=0)

    def test_profiles_reject_changed_policy_even_with_a_new_checksum(self):
        from backend.mirror_profile import digest
        for blend in [False,True]:
            p=profile(blend);self.assertEqual(validate_profile(p),p)
            p['threshold']=.15;p['modelHash']=digest({k:v for k,v in p.items() if k!='modelHash'})
            with self.assertRaises(ValueError):validate_profile(p)

    def test_blend_preserves_frozen_or_boundaries(self):
        class Component:
            def __init__(self,p,logit,key):self.profile=p;self.logit=logit;self.key=key;self.device=torch.device('cpu')
            def predict(self,data,variant):return {'elapsedMs':0,'logit':self.logit,'score':sigmoid(self.logit),'components':{self.key:{'logit':self.logit,'score':sigmoid(self.logit)}}}
        for a,b in [(-3,-2),(0,-2),(-2,0),(1,2)]:
            e=MirrorCommunityEngine(Component(profile(),a,'mirror'),Component({'modelHash':profile(True)['communityHash']},b,'community'))
            r=e.predict(b'contract stub')
            self.assertEqual(r['score']>=.5,a>=0 or b>=0);self.assertEqual(r['logit'],max(a,b))

    def test_retired_api_models_rejected_before_loading(self):
        from backend.app import app
        from backend.tests.test_backend import png
        client=TestClient(app)
        for model in ['ferretnet-b-v1','fiser-dear-r-v1']:
            r=client.post('/api/detect',files={'file':('x.png',png(),'image/png')},data={'model':model})
            self.assertEqual(r.status_code,422);self.assertIn('retired',r.json()['detail'])

if __name__=='__main__':unittest.main()

class FrozenMemoryTests(unittest.TestCase):
    def test_cached_keys_and_values_preserve_outputs_and_reuse_projections(self):
        from backend.mirror import CachedMemoryBank
        from backend.vendor.mirror.upstream import MirrorMemoryBank
        bank=CachedMemoryBank(16).eval()
        x=torch.randn(1,9,16)
        with torch.inference_mode():
            reference=MirrorMemoryBank.forward(bank,x)
            with patch.object(bank.k_proj,'forward',wraps=bank.k_proj.forward) as projection:
                first=bank(x);second=bank(x)
                self.assertEqual(projection.call_count,1)
            for actual,expected in zip(first,reference):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for actual,expected in zip(second,reference):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            bank.to('cpu');self.assertIsNone(bank._cached_k)
