import unittest
from benchmark_mcp.worker import metrics

class BenchmarkMetricsTests(unittest.TestCase):
    def test_confusion_and_tied_score_ranking(self):
        rows=[{'label':label,'group':'x','score':score} for label,score in
              [('ai',.9),('ai',.5),('real',.5),('real',.1)]]
        result=metrics(rows,.5)
        self.assertEqual([result[k] for k in ['tp','fp','tn','fn']],[2,1,1,0])
        self.assertAlmostEqual(result['precision'],2/3)
        self.assertAlmostEqual(result['recall'],1)
        self.assertAlmostEqual(result['f1'],.8)
        self.assertAlmostEqual(result['auroc'],.875)
        self.assertEqual(result['categories']['real/x']['flagRate'],.5)

    def test_no_flags_or_single_class_do_not_fabricate_auc(self):
        result=metrics([{'label':'real','group':'photo','score':.1}],.5)
        self.assertIsNone(result['auroc'])
        self.assertIsNone(result['averagePrecision'])
        self.assertEqual(result['precision'],0)
        self.assertEqual(result['fpr'],0)

if __name__=='__main__':
    unittest.main()
