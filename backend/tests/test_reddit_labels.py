import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('reddit_labels',Path(__file__).resolve().parents[2]/'datasets/reddit/labels.py')
labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)

class ProvenanceTests(unittest.TestCase):
    def test_unambiguous_answers_and_negative_ai_are_handled(self):
        for text,want in [('The answer is AI - generated with Gemini','ai'),('The answer is real','real'),('The answer is not AI','real'),('Not AI. I took this photo.','real'),('generated in Midjourney','ai')]:
            self.assertEqual(labels.classify_answer(text)[0],want,text)

    def test_mixed_galleries_and_edited_answers_are_not_training_truth(self):
        for text in ['The answer is AI but one is AI and the other is real','The answer is real, with an AI edited background','The first image is real and the second image is AI','Both are real','The answer is AI. '+('details '*60)+' Partially generated']:
            self.assertIsNone(labels.classify_answer(text)[0],text)

    def test_only_marked_op_answer_is_used(self):
        self.assertIsNone(labels.realorai_label('I think this looks AI')[0])
        self.assertEqual(labels.realorai_label("Here's the answer I got from OP: &gt;!The answer is real!&lt;")[0],'real')

    def test_gallery_panel_needs_its_own_evidence(self):
        r={'basis':'isthisai-solved','basisNote':'Solved [AI]','label':'ai','galleryIndex':2}
        self.assertTrue(labels.needs_review(r))

if __name__=='__main__':unittest.main()
