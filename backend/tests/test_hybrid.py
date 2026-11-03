import io
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient

from backend.hybrid import HybridEngine, PROFILE, centered_fusion, preprocess, MODEL_ID, MODEL_HASH
from backend.app import app
from backend.vendor.opensynthid import preprocess as upstream_preprocess


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=HybridEngine()

    def test_fusion_preserves_the_frozen_or_rule(self):
        cuts=PROFILE['componentThresholds']
        for f in [-1000,-15,0,1,2,15,1000]:
            for o in [-1000,-15,0,1,2,15,1000]:
                score,logit=centered_fusion(f,o)
                expected=f>=math.log(cuts['ferretnet']/(1-cuts['ferretnet'])) or o>=math.log(cuts['opensynthid']/(1-cuts['opensynthid']))
                self.assertEqual(score>=.5,expected)
                self.assertTrue(math.isfinite(logit))
        for f,o in [(float('nan'),0),(0,float('inf'))]:
            with self.assertRaises(ValueError):centered_fusion(f,o)

    def test_six_channel_preprocessing_matches_the_upstream(self):
        import tempfile
        rng=np.random.default_rng(20261011)
        im=rng.integers(0,256,(333,481,3),dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            Image.fromarray(im).save(f.name)
            self.assertTrue(torch.equal(preprocess(im),upstream_preprocess(f.name)))

    def test_actual_model_and_api_return_both_signals(self):
        stream=io.BytesIO()
        Image.new('RGB',(320,240),(30,90,180)).save(stream,'PNG')
        for variant in ['original','jpeg75','resize512']:
            result=self.engine.predict(stream.getvalue(),variant)
            self.assertEqual(result['model'],MODEL_ID)
            self.assertEqual(result['modelHash'],MODEL_HASH)
            self.assertFalse(result['watermarkVerified'])
            self.assertEqual(set(result['components']),{'ferretnet_b','opensynthid'})
            self.assertEqual((result['width'],result['height']),(320,240))
            with patch('backend.app.engine',self.engine.ferret),patch('backend.app.hybrid_engine',self.engine):
                response=TestClient(app).post('/api/detect',files={'file':('fixture.png',stream.getvalue(),'image/png')},data={'model':MODEL_ID,'variant':variant})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['score'],result['score'])

    def test_checkpoint_corruption_is_rejected(self):
        with patch('backend.hybrid.sha256',return_value='bad'):
            with self.assertRaisesRegex(RuntimeError,'checksum'):HybridEngine(self.engine.ferret)


if __name__=='__main__':unittest.main()
