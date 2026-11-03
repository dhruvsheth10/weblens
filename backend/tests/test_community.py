import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.assets import MODELS
from backend.community import CommunityEngine, CommunityFerretEngine, validate_profile, canonical_hash
from backend.ferretnet import FerretNetEngine, THRESHOLD
from backend.image_io import sigmoid
from backend.previews import predict_image
from backend.tests.test_backend import png

PROJECT=Path(__file__).resolve().parents[2]

class CommunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=CommunityEngine()

    def test_fitted_head_matches_saved_training_logits(self):
        report=json.loads((PROJECT/'artifacts/semantic-preview-2026-10-11/training-report.json').read_text())
        root=Path('/Users/dhruvsheth/Downloads/benchmark/calibration')
        # Explicit development-only paths, never enumerate another partition.
        for row in report['parityExamples']:
            relative=Path(row['file']);self.assertEqual(relative.parts[0],'calibration')
            path=root/Path(*relative.parts[1:])
            self.assertTrue(path.resolve().is_relative_to(root))
            data=path.read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),row['sourceHash'])
            result=self.engine.predict(data)
            self.assertAlmostEqual(result['logit'],row['expectedLogit'],places=4)
            self.assertAlmostEqual(result['score'],sigmoid(result['logit']),places=12)

    def test_all_supported_variants_return_actual_local_model_outputs(self):
        for variant in ['original','jpeg75','resize512','reddit-webp640','reddit-webp960']:
            result=predict_image(self.engine,png((1024,300)),variant)
            self.assertEqual(result['variant'],variant);self.assertEqual(result['width'],1024)
            self.assertIn('community_forensics',result['components'])
            self.assertFalse(result['watermarkVerified'])
            self.assertTrue(np.isfinite(result['logit']))

    def test_corrupt_weights_and_profiles_fail_without_prediction(self):
        with patch('backend.community.sha256',return_value='wrong'):
            with self.assertRaises(RuntimeError):CommunityEngine()
        for patch_value in [{'head':{'weight':[0]*383,'bias':0}},{'threshold':.6},{'model':'python-exec'},{'uncertainThreshold':float('nan')}]:
            profile={**copy.deepcopy(self.engine.profile),**patch_value}
            if not np.isnan(profile['uncertainThreshold']):profile['modelHash']=canonical_hash(profile)
            with self.assertRaises(ValueError):validate_profile(profile)

    def test_api_uses_same_frozen_profile_and_preview_variant(self):
        from backend.app import app
        with patch('backend.app.community_engine',self.engine):
            result=TestClient(app).post('/api/detect',files={'file':('test.png',png((1280,720)),'image/png')},data={'model':self.engine.profile['model'],'variant':'reddit-webp640'})
            self.assertEqual(result.status_code,200,result.text)
            self.assertEqual(result.json()['modelHash'],self.engine.profile['modelHash'])
            self.assertEqual(result.json()['transformation']['size'],[640,360])

    def test_combination_exactly_preserves_component_high_decisions_and_shares_decode(self):
        combined=CommunityFerretEngine(self.engine,FerretNetEngine())
        data=png((1280,300))
        for variant in ['original','reddit-webp640','reddit-webp960']:
            semantic=predict_image(self.engine,data,variant)
            ferret=predict_image(combined.ferret,data,variant)
            with patch('backend.community.decode_image',wraps=__import__('backend.image_io',fromlist=['decode_image']).decode_image) as decoder:
                result=predict_image(combined,data,variant)
                self.assertEqual(decoder.call_count,1)
            self.assertEqual(result['score']>=.5,semantic['score']>=.5 or ferret['score']>=THRESHOLD)
            self.assertEqual(result['semanticScore'],semantic['score'])
            self.assertAlmostEqual(result['components']['ferretnet_b']['logit'],ferret['logit'],places=6)
            self.assertAlmostEqual(result['components']['community_forensics']['logit'],semantic['logit'],places=6)
        changed=copy.deepcopy(combined.profile);changed['semanticProfileHash']='0'*64;changed['modelHash']=canonical_hash(changed)
        with self.assertRaises(ValueError):CommunityFerretEngine(self.engine,combined.ferret,changed)

if __name__=='__main__':unittest.main()
