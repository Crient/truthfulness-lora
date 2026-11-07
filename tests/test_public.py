"""Dependency-light tests using invented examples, never benchmark records."""
import math
from pathlib import Path
import tempfile
import unittest
from truthfulness_lora.metrics import metric, summarize, ARMS
from truthfulness_lora.runtime import claim_output

class PublicTests(unittest.TestCase):
    def test_metrics_and_tie_credit(self):
        t={'choices':['synthetic-positive','synthetic-negative'],'labels':[1,0]}
        self.assertEqual(metric(dict(zip(t['choices'],[0.,0.])),t,'mc1'),.5)
        self.assertAlmostEqual(metric(dict(zip(t['choices'],[0.,-math.log(3)])),t,'mc2'),.75)
        self.assertEqual(metric(dict(zip(t['choices'],[-2.,-1.])),t,'mc1'),0.)

    def test_stable_softmax_and_order(self):
        a={'choices':['one','two'],'labels':[0,1]};b={'choices':['two','one'],'labels':[1,0]}
        s={'one':-10000.,'two':-10001.}
        self.assertAlmostEqual(metric(s,a,'mc2'),metric(s,b,'mc2'))
        self.assertAlmostEqual(metric(s,a,'mc2'),metric({k:v+9999 for k,v in s.items()},a,'mc2'))

    def test_pairing_and_reproducibility(self):
        records={a:[{'benchmark_index':i,'question':f'synthetic-{i}','mc1':float(i%2),'mc2':i/10} for i in range(5)] for a in ARMS}
        x=summarize(records,resamples=32)
        self.assertEqual(x,summarize(records,resamples=32))
        self.assertEqual(x['paired_differences']['DPO-minus-SFT']['mc2']['mean'],0)
        records['DPO'].reverse()
        with self.assertRaises(ValueError):summarize(records,resamples=32)

    def test_no_overwrite_or_nested_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir()
            out=claim_output(root/'new',[src]);(out/'sentinel').write_text('untouched')
            with self.assertRaises(FileExistsError):claim_output(out)
            with self.assertRaises(ValueError):claim_output(src/'child',[src])
            with self.assertRaises(ValueError):claim_output(root,[src])
            self.assertEqual((out/'sentinel').read_text(),'untouched')

if __name__=='__main__':unittest.main()
