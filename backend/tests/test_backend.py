import io
import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.engine import combine, decode_image, native_regions, sigmoid
from backend.ferretnet import FerretNetEngine, MODEL_HASH as FERRETNET_HASH, MODEL_ID as FERRETNET_MODEL_ID


def png(size=(320, 240)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 180, 80)).save(buffer, "PNG")
    return buffer.getvalue()


class ImageContractTests(unittest.TestCase):
    def test_invalid_images_and_unsupported_variants_are_rejected(self):
        for data, variant in ((b"not an image", "original"), (png((64, 64)), "original"), (png(), "unknown")):
            with self.assertRaises(ValueError):
                decode_image(data, variant)

    def test_resilience_variants_keep_original_dimensions_in_result(self):
        image, source_size = decode_image(png((1024, 768)), "resize512")
        self.assertEqual(source_size, (1024, 768))
        self.assertEqual(image.size, (512, 384))
        jpeg, source_size = decode_image(png((1024, 768)), "jpeg75")
        self.assertEqual(jpeg.size, source_size)

    def test_native_regions_cover_standard_images_without_downsampling(self):
        regions = native_regions(1024, 1024)
        self.assertEqual(len(regions), 4)
        self.assertIn((0, 0, 768, 768), regions)
        self.assertIn((256, 256, 1024, 1024), regions)
        self.assertEqual(native_regions(320, 240), [(0, 0, 320, 240)])
        self.assertLessEqual(len(native_regions(6000, 6000)), 16)

    def test_both_models_contribute_and_score_logit_are_consistent(self):
        calibration = {"fiser": {"a": 1, "b": 0}, "dear_r": {"a": 1, "b": 0}}
        baseline, _, _ = combine(-2, -2, calibration)
        fiser, _, _ = combine(2, -2, calibration)
        dear, _, _ = combine(-2, 2, calibration)
        self.assertGreater(fiser, baseline)
        self.assertGreater(dear, baseline)
        score, logit, _ = combine(1000, 1000, calibration)
        self.assertTrue(math.isfinite(logit))
        self.assertAlmostEqual(sigmoid(logit), score)
        with self.assertRaises(ValueError):
            combine(float("nan"), 1, calibration)


class ApiContractTests(unittest.TestCase):
    def test_health_publishes_pinned_identity_for_every_picker_model(self):
        from types import SimpleNamespace
        from pathlib import Path
        import json
        with patch('backend.app.engine',SimpleNamespace(device=SimpleNamespace(type='cpu'))):
            available=TestClient(app).get('/api/health').json()['availableModels']
        for name in ['community-ferret','community','hybrid','mirror','mirror-community','reddit-student']:
            profile=json.loads((Path(__file__).parents[1]/'models'/f'{name}-profile.json').read_text())
            for key in ['modelHash','pipeline','threshold']:
                self.assertEqual(available[profile['model']][key],profile[key])

    def test_student_is_loaded_once_only_when_selected(self):
        import asyncio
        from backend.app import STUDENT_PROFILE, selected_engine
        sentinel = object()
        with patch('backend.app.student_engine', None), patch('backend.reddit_research.RedditStudentEngine', return_value=sentinel) as constructor:
            async def select_twice():
                self.assertIs(await selected_engine(STUDENT_PROFILE['model']), sentinel)
                self.assertIs(await selected_engine(STUDENT_PROFILE['model']), sentinel)
            asyncio.run(select_twice())
            constructor.assert_called_once_with(STUDENT_PROFILE)

    def test_bad_file_returns_actionable_validation_error(self):
        class Decoder:
            def predict(self, data, variant):
                decode_image(data, variant)
        with patch("backend.app.engine", Decoder()):
            client = TestClient(app)
            response = client.post("/api/detect", files={"file": ("bad.png", b"not png", "image/png")})
            self.assertEqual(response.status_code, 422)
            self.assertIn("decoded", response.json()["detail"])

    def test_backend_unavailable_never_returns_a_fake_prediction(self):
        with patch("backend.app.engine", None):
            client = TestClient(app)
            self.assertEqual(client.get("/api/health").status_code, 503)
            self.assertEqual(client.post("/api/detect", files={"file": ("x.png", png(), "image/png")}, data={"model":"community-ferret-preview-v1"}).status_code, 503)


class FerretNetContractTests(unittest.TestCase):
    def test_released_checkpoint_returns_a_valid_local_detection(self):
        result = FerretNetEngine().predict(png((320, 320)))
        self.assertEqual(result["model"], FERRETNET_MODEL_ID)
        self.assertEqual(result["modelHash"], FERRETNET_HASH)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)
        self.assertAlmostEqual(sigmoid(result["logit"]), result["score"])
        self.assertEqual(result["components"]["ferretnet_b"]["crop"], [256, 256])


if __name__ == "__main__":
    unittest.main()
