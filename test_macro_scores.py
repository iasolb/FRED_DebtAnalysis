import unittest

import pandas as pd

from macro_scores import _attach_continuous


class TaylorRuleRStarTests(unittest.TestCase):
    def test_uses_hlw_r_star_before_sep_and_sep_when_available(self):
        df = pd.DataFrame(
            {
                "SEP_r_star": [float("nan"), 0.5],
                "HLW_r_star": [2.0, 1.0],
                "Core_PCE_YoY": [2.0, 2.0],
                "Unemployment_Rate": [4.0, 4.0],
                "FedFunds_Rate": [0.0, 0.0],
                "Policy_Stance_SEP": [0.0, 0.0],
                "Deficit_GDP_Pct": [0.0, 0.0],
                "UMich_Inflation_Expectations": [2.0, 2.0],
                "Breakeven_Inflation_5Y": [2.0, 2.0],
                "Breakeven_Inflation_10Y": [2.0, 2.0],
                "Chicago_Fed_Financial_Conditions": [0.0, 0.0],
                "Consumer_Credit_Total": [1.0, 1.0],
                "Case_Shiller_Home_Price": [1.0, 1.0],
                "Mortgage_Rate_30Y": [1.0, 1.0],
            }
        )

        scored = _attach_continuous(df)

        self.assertEqual(scored["Taylor_Prescribed"].tolist(), [4.0, 2.5])


if __name__ == "__main__":
    unittest.main()
