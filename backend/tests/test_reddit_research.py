import copy,unittest
import torch
from backend.reddit_research import RedditFusionEngine,RedditStudentEngine
from backend.reddit_research_profile import validate_profile,digest
from backend.mirror_profile import profile as mirror_profile,COMMUNITY_SHA
from backend.semantic_profile import CHECKPOINT_SHA

def fusion_profile():
 p={'schemaVersion':1,'kind':'reddit-fusion','model':'reddit-fusion-test-v1','pipeline':'clipped-component-logits-standardized-logistic-v1','components':['community_forensics','ferretnet_b','mirror'],'selectedFeatures':[0,2],'clip':20.,'weight':[.3,.7],'bias':-.2,'threshold':.5,'mirrorHash':mirror_profile()['modelHash'],'communityHash':COMMUNITY_SHA};p['modelHash']=digest(p);return p

def student_profile():
 p={'schemaVersion':1,'kind':'reddit-student','model':'reddit-student-test-v1','pipeline':'community-vit-s224-supervised-last2-v1','checkpointSha256':'a'*64,'sourceCheckpointSha256':CHECKPOINT_SHA,'checkpointPath':'artifacts/reddit-adaptation-2026-10-14/models/student-'+('a'*64)+'.safetensors','logitOffset':2.,'threshold':.5,'trainingHash':'b'*64};p['modelHash']=digest(p);return p

class ResearchProfileTests(unittest.TestCase):
 def test_numeric_fusion_is_pinned_and_matches_runtime_dot_product(self):
  p=fusion_profile();self.assertEqual(validate_profile(p),p)
  class Combination:
   device=torch.device('cpu')
   def predict(self,data,variant):return {'logit':0,'score':.5,'components':{'community_forensics':{'logit':100.},'ferretnet_b':{'logit':3.},'mirror':{'logit':-2.}}}
  r=RedditFusionEngine(p,Combination()).predict(b'unit fixture')
  self.assertAlmostEqual(r['logit'],20*.3-2*.7-.2);self.assertEqual(r['modelHash'],p['modelHash'])
 def test_corrupt_or_executable_representations_are_rejected(self):
  for change in [{'clip':5.},{'selectedFeatures':[False,2]},{'weight':[float('inf'),1.]},{'components':['filename','subreddit','mirror']},{'mirrorHash':'0'*64}]:
   p={**fusion_profile(),**change}
   try:p['modelHash']=digest(p)
   except ValueError:continue
   with self.assertRaises(ValueError):validate_profile(p)
 def test_student_path_is_bounded_to_content_addressed_weights(self):
  p=student_profile();self.assertEqual(validate_profile(p),p)
  with self.assertRaises(ValueError):RedditStudentEngine(p)
  for path in ['/tmp/model.safetensors','artifacts/../backend/model.safetensors','backend/engine.py',None,123]:
   changed={**p,'checkpointPath':path};changed['modelHash']=digest(changed)
   with self.assertRaises(ValueError):validate_profile(changed)
if __name__=='__main__':unittest.main()
