import io
import unittest
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T
from backend.assets import FERRETNET_SHA
from backend.ferret_adapted import crop_regions, profile_hash, validate_profile, FerretNetAdaptedEngine
from backend.image_io import decode_image
from backend.engine import decode_image as legacy_decode
from benchmark_mcp.core import validate_profile as mcp_validate

def encoded(image,format='PNG',**kwargs):
    stream=io.BytesIO();image.save(stream,format,**kwargs);return stream.getvalue()

def profile():
    value={'schemaVersion':1,'checkpointSha256':FERRETNET_SHA,'kind':'probe','representation':'center256',
           'head':{'weight':[0.0]*384,'bias':1.0},'model':'ferretnet-test-probe-v1','pipeline':'test-probe-v1','threshold':.5}
    value['modelHash']=profile_hash(value)
    return value

class FerretProfileTests(unittest.TestCase):
    def test_crop_before_tensor_preserves_original_rounding_and_black_padding(self):
        rng=np.random.default_rng(721)
        for width,height in [(128,128),(255,257),(257,255),(511,513),(260,128),(1024,768)]:
            image=Image.fromarray(rng.integers(0,256,(height,width,3),dtype=np.uint8))
            reference=T.CenterCrop(256)(T.ToImage()(image))
            actual=T.ToImage()(crop_regions(image,256,1)[0])
            self.assertTrue(torch.equal(reference,actual),(width,height))

    def test_standard_decoding_matches_pinned_legacy_pipeline(self):
        image=Image.new('RGBA',(513,257),(123,56,200,140))
        for data in (encoded(image),encoded(image.convert('RGB'),'JPEG'),encoded(image,'WEBP')):
            for variant in ('original','jpeg75','resize512'):
                new,new_size=decode_image(data,variant)
                old,old_size=legacy_decode(data,variant)
                self.assertEqual(new_size,old_size)
                self.assertTrue(np.array_equal(new,old))

    def test_mpo_uses_primary_jpeg_frame(self):
        primary=Image.new('RGB',(320,240),(100,200,30))
        data=encoded(primary,'MPO',save_all=True,append_images=[Image.new('RGB',(320,240),(30,20,10))])
        decoded,size=decode_image(data)
        self.assertEqual(size,primary.size)
        self.assertLess(abs(decoded.getpixel((50,50))[1]-200),3)

    def test_profile_hash_and_validation_agree_with_mcp(self):
        value=profile()
        validate_profile(value);mcp_validate(value)
        for change in ({'threshold':float('nan')},{'head':{'weight':[0]*384,'bias':True}},
                       {'head':{'weight':[0]*383,'bias':0}}):
            invalid={**value,**change}
            for validator in (validate_profile,mcp_validate):
                with self.assertRaises(ValueError): validator(invalid)
        changed={**value,'threshold':.6}
        with self.assertRaisesRegex(ValueError,'checksum'): validate_profile(changed)

    def test_real_probe_forward_and_result_identity(self):
        value=profile()
        result=FerretNetAdaptedEngine(profile=value).predict(encoded(Image.new('RGB',(320,320),(100,20,200))))
        self.assertEqual(result['modelHash'],value['modelHash'])
        self.assertEqual(result['logit'],1.0)
        self.assertAlmostEqual(result['score'],1/(1+np.exp(-1)))
        self.assertEqual(result['components']['ferretnet_adapted']['crops'],1)

if __name__=='__main__': unittest.main()
