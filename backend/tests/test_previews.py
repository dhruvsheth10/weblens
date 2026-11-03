import io
import unittest
import numpy as np
from PIL import Image
from backend.image_io import decode_image
from backend.previews import preview_bytes, predict_image

def png(size):
    x,y=np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    image=Image.fromarray(np.stack([(x*3+y)%256,(y*7)%256,(x+y*5)%256],axis=-1).astype('uint8'))
    stream=io.BytesIO();image.save(stream,'PNG');return stream.getvalue()

class PreviewTests(unittest.TestCase):
    def test_one_codec_pass_and_exact_aspect_ratio(self):
        data=png((1200,900))
        for variant,size in [('reddit-webp640',(640,480)),('reddit-webp960',(960,720))]:
            encoded,original,actual=preview_bytes(data,variant)
            self.assertEqual(original,(1200,900));self.assertEqual(actual,size)
            transformed,source=decode_image(data,variant)
            with Image.open(io.BytesIO(encoded)) as opened:
                self.assertEqual(opened.format,'WEBP')
                np.testing.assert_array_equal(np.asarray(transformed),np.asarray(opened.convert('RGB')))
            self.assertEqual(source,(1200,900))

    def test_no_upscale_and_validated_narrow_previews(self):
        self.assertEqual(decode_image(png((200,180)),'reddit-webp640')[0].size,(200,180))
        image,source=decode_image(png((2000,200)),'reddit-webp640')
        self.assertEqual(image.size,(640,64));self.assertEqual(source,(2000,200))
        with self.assertRaises(ValueError):decode_image(png((2000,64)),'reddit-webp640')

    def test_prediction_records_simulation_and_preserves_original_dimensions(self):
        class Engine:
            def predict(self,data,variant):
                image,source=decode_image(data,variant)
                return {'width':source[0],'height':source[1],'variant':variant,'score':.1,'actual':image.size}
        result=predict_image(Engine(),png((2000,200)),'reddit-webp640')
        self.assertEqual(result['actual'],(640,64));self.assertEqual(result['width'],2000)
        self.assertTrue(result['transformation']['simulation'])
        with self.assertRaises(ValueError):predict_image(Engine(),png((300,300)),'validation')

if __name__=='__main__':unittest.main()
