import unittest
import numpy as np
import pandas as pd
from energy import generate_data, clean_data, make_features

class EnergyTests(unittest.TestCase):
    def test_generation(self):
        raw = generate_data(days=12)
        self.assertEqual(len(raw),12*24*5)
        self.assertEqual(raw.building.nunique(),5)
    def test_past_only_lag(self):
        clean,_ = clean_data(generate_data(days=12))
        f = make_features(clean)
        b = f[f.building==f.building.iloc[0]].reset_index(drop=True)
        self.assertAlmostEqual(b.lag_1.iloc[10],b.energy_kwh.iloc[9])
        self.assertTrue(np.isnan(b.lag_168.iloc[167]))
    def test_clean_invalid_duplicate(self):
        raw = pd.DataFrame({'timestamp':['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 01:00','bad'],'building':['A']*4,'energy_kwh':[10,20,-2,4]})
        c,q = clean_data(raw)
        self.assertEqual(q['invalid_keys_removed'],1)
        self.assertEqual(q['duplicate_rows_removed'],1)
        self.assertFalse(c.energy_observed.iloc[-1])
        self.assertEqual(c.energy_kwh.iloc[-1],10)

if __name__ == '__main__':
    unittest.main()
