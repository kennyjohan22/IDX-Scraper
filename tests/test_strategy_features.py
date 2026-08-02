import unittest

import pandas as pd

from strategy_features import obv_slope_is_positive, position_in_trading_range


class PositionInTradingRangeTests(unittest.TestCase):
    def test_uses_only_latest_252_observations_and_ignores_zero_ohlc(self):
        history = pd.DataFrame(
            {
                "Tertinggi": [1_000] * 48 + [200] * 251 + [0],
                "Terendah": [1] * 48 + [100] * 251 + [0],
                "Penutupan": [500] * 48 + [150] * 252,
            }
        )

        position = position_in_trading_range(history, as_of_pos=len(history))

        self.assertEqual(position, 50.0)

    def test_returns_neutral_position_when_no_valid_range_exists(self):
        history = pd.DataFrame(
            {"Tertinggi": [0, 100], "Terendah": [0, 100], "Penutupan": [100, 100]}
        )

        position = position_in_trading_range(history, as_of_pos=len(history))

        self.assertEqual(position, 50.0)


class ObvSlopeTests(unittest.TestCase):
    def test_detects_falling_obv_even_when_final_balance_is_positive(self):
        signal = pd.DataFrame(
            {
                "Sebelumnya": [100, 101, 100],
                "Penutupan": [101, 100, 99],
                "Volume": [1_000, 1, 1],
            }
        )

        self.assertFalse(obv_slope_is_positive(signal))

if __name__ == "__main__":
    unittest.main()
