import unittest
import numpy as np
from research.reddit_adaptation.common import cutoff_at_category_fpr,summarize
from research.reddit_adaptation.fusion import fit,weighted
class AdaptationTests(unittest.TestCase):
 def test_category_cap_uses_strict_tie_boundary(self):
  logits=np.array([.2,.2,.7,2.]);labels=[0,0,0,1];categories=['photo','photo','art','art']
  cut=cutoff_at_category_fpr(logits,labels,categories)
  self.assertGreater(cut,.7);self.assertEqual(sum(logits[:3]>=cut),0)
 def test_one_percent_budget_is_per_category(self):
  logits=np.arange(200,dtype=float);labels=np.zeros(200);categories=['photo']*100+['art']*100
  cut=cutoff_at_category_fpr(logits,labels,categories)
  self.assertEqual(sum(logits[100:]>=cut),1);self.assertEqual(sum(logits[:100]>=cut),0)
 def test_precision_is_reweighted_for_prevalence(self):
  rows=[{'label':'ai','category':'photo'}]*2+[{'label':'real','category':'photo'}]*8
  metric=summarize([1,1,1,1,-1,-1,-1,-1,-1,-1],rows)
  self.assertEqual(metric['precision'],.5);self.assertLess(metric['precisionAtAssumedPrevalence']['0.05'],.2)
 def test_source_group_weights_do_not_multiply_gallery_influence(self):
  rows=[{'label':'ai','source':'x','sourceGroup':'a'}]*4+[{'label':'ai','source':'x','sourceGroup':'b'}]+[{'label':'real','source':'y','sourceGroup':'c'}]
  w=weighted(rows);self.assertAlmostEqual(sum(w[:4]),w[4]);self.assertAlmostEqual(sum(w[:5]),w[5])
 def test_fold_local_standardization_folds_into_runtime_dot_product(self):
  rows=[{'label':'ai' if i%2 else 'real','source':'a' if i%2 else 'b','sourceGroup':str(i)} for i in range(20)]
  x=np.array([[i%2*2-1,i/5] for i in range(20)],dtype=float);y=np.array([i%2 for i in range(20)])
  w,b=fit(x,y,rows,.1);logits=x@w+b;self.assertTrue(np.isfinite(logits).all());self.assertGreater(logits[y==1].mean(),logits[y==0].mean())
if __name__=='__main__':unittest.main()

class PairCandidateTests(unittest.TestCase):
 def test_original_candidates_use_media_identity_and_safe_host(self):
  import importlib.util
  from pathlib import Path
  spec=importlib.util.spec_from_file_location('pair_backfill',Path(__file__).resolve().parents[2]/'scripts/backfill-reddit-development-pairs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  row={'preview':{'url':'https://preview.redd.it/some-title-v0-abc123def456.jpeg?width=640&format=png'},'original':None}
  urls=module.candidates(row);self.assertEqual(urls[0],'https://i.redd.it/abc123def456.jpeg');self.assertIn('https://i.redd.it/abc123def456.png',urls)
  row['original']={'url':'https://example.com/private.jpg'};self.assertTrue(all(u.startswith('https://i.redd.it/') for u in module.candidates(row)))
