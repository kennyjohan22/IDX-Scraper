import unittest

import numpy as np
import pandas as pd

from weekly_strategy import (
    assess_weekly_regime,
    build_weekly_trade_plan,
    calculate_position_size,
    compute_weekly_factors,
    estimate_foreign_value,
    idx_tick_size,
    rank_weekly_candidates,
    round_to_idx_tick,
    select_actionable_candidates,
)


def make_history(ticker="TEST", periods=60, start=100.0, step=1.0):
    dates = pd.bdate_range("2026-05-11", periods=periods)
    closes = start + np.arange(periods) * step
    previous = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(
        {
            "DATE": dates,
            "Kode Saham": ticker,
            "Sebelumnya": previous,
            "Open Price": closes - 0.5,
            "Tertinggi": closes + 2.0,
            "Terendah": closes - 2.0,
            "Penutupan": closes,
            "Volume": np.full(periods, 5_000_000.0),
            "Nilai": closes * 5_000_000.0,
            "Frekuensi": np.full(periods, 2_000.0),
            "Foreign Buy": np.full(periods, 1_200_000.0),
            "Foreign Sell": np.full(periods, 800_000.0),
        }
    )


class WeeklyFactorTests(unittest.TestCase):
    def test_computes_relative_strength_trend_quality_and_close_location(self):
        history = make_history()

        factors = compute_weekly_factors(
            history,
            as_of_date=history["DATE"].max(),
            market_return_20d_pct=5.0,
        )

        expected_return = (history.iloc[-1]["Penutupan"] / history.iloc[-21]["Penutupan"] - 1) * 100
        self.assertAlmostEqual(factors["return_20d_pct"], expected_return)
        self.assertAlmostEqual(
            factors["relative_strength_20d_pct"], expected_return - 5.0
        )
        self.assertGreater(factors["trend_efficiency_20d"], 0.95)
        self.assertAlmostEqual(factors["close_location_5d"], 0.5)
        self.assertTrue(factors["latest_active"])

    def test_marks_a_ticker_stale_when_it_did_not_trade_on_the_cutoff(self):
        history = make_history()
        history.loc[history.index[-1], "Volume"] = 0

        factors = compute_weekly_factors(
            history,
            as_of_date=history["DATE"].max(),
            market_return_20d_pct=0.0,
        )

        self.assertFalse(factors["latest_active"])


class PositionSizingTests(unittest.TestCase):
    def test_idx_tick_rounding_uses_exchange_price_bands(self):
        self.assertEqual(
            [idx_tick_size(price) for price in [199, 200, 500, 2_000, 5_000]],
            [1, 2, 5, 10, 25],
        )
        self.assertEqual(round_to_idx_tick(6_408, direction="down"), 6_400)
        self.assertEqual(round_to_idx_tick(6_408, direction="up"), 6_425)

    def test_caps_position_by_available_capital_even_when_stop_is_tight(self):
        sizing = calculate_position_size(
            entry=100,
            stop=99,
            account_size=1_000_000,
            risk_pct=1.0,
            max_allocation_pct=20.0,
        )

        self.assertEqual(sizing["lots"], 20)
        self.assertEqual(sizing["shares"], 2_000)
        self.assertEqual(sizing["capital"], 200_000)
        self.assertEqual(sizing["risk_idr"], 2_000)


class ForeignUnitTests(unittest.TestCase):
    def test_estimates_value_from_foreign_share_volume_and_close(self):
        estimated = estimate_foreign_value(
            pd.Series([100.0, -50.0]), pd.Series([1_000.0, 2_000.0])
        )

        self.assertEqual(estimated.tolist(), [100_000.0, -100_000.0])


class WeeklyRegimeTests(unittest.TestCase):
    def test_uses_current_liquid_names_and_detects_broad_risk_on_trend(self):
        histories = [make_history(ticker=f"T{i}", step=1.0 + i * 0.05) for i in range(8)]
        market = pd.concat(histories, ignore_index=True)

        regime = assess_weekly_regime(market, as_of_date=market["DATE"].max())

        self.assertEqual(regime["label"], "RISK-ON")
        self.assertEqual(regime["liquid_stocks"], 8)
        self.assertGreater(regime["pct_above_ma20"], 90)
        self.assertGreater(regime["median_return_20d_pct"], 0)


class WeeklyRankingTests(unittest.TestCase):
    def test_excludes_tickers_with_missing_latest_price_or_volume(self):
        good = make_history(ticker="GOOD", periods=80, start=500)
        missing_close = make_history(ticker="NO_CLOSE", periods=80, start=500)
        missing_volume = make_history(ticker="NO_VOLUME", periods=80, start=500)
        missing_close.loc[missing_close.index[-1], "Penutupan"] = np.nan
        missing_volume.loc[missing_volume.index[-1], "Volume"] = np.nan

        ranked = rank_weekly_candidates(
            pd.concat([good, missing_close, missing_volume], ignore_index=True),
            as_of_date=good["DATE"].max(),
            min_avg_value=100_000_000,
        )

        self.assertEqual(ranked["Ticker"].tolist(), ["GOOD"])

    def test_excludes_ticker_when_recent_open_prices_are_missing_zeros(self):
        good = make_history(ticker="GOOD", periods=80, start=500)
        zero_open = make_history(ticker="ZERO_OPEN", periods=80, start=500)
        zero_open.loc[zero_open.index[-5:], "Open Price"] = 0.0

        ranked = rank_weekly_candidates(
            pd.concat([good, zero_open], ignore_index=True),
            as_of_date=good["DATE"].max(),
            min_avg_value=100_000_000,
        )

        self.assertEqual(ranked["Ticker"].tolist(), ["GOOD"])

    def test_absolute_gap_penalizes_large_gap_downs_instead_of_rewarding_them(self):
        steady = make_history(ticker="STEADY_OPEN", periods=80, start=500)
        gap_down = make_history(ticker="GAP_DOWN", periods=80, start=500)
        steady.loc[steady.index[-5:], "Open Price"] = steady.loc[
            steady.index[-5:], "Sebelumnya"
        ]
        gap_down.loc[gap_down.index[-5:], "Open Price"] = (
            gap_down.loc[gap_down.index[-5:], "Sebelumnya"] * 0.80
        )

        ranked = rank_weekly_candidates(
            pd.concat([steady, gap_down], ignore_index=True),
            as_of_date=steady["DATE"].max(),
            min_avg_value=100_000_000,
        ).set_index("Ticker")

        self.assertGreater(
            ranked.loc["STEADY_OPEN", "Weekly Score"],
            ranked.loc["GAP_DOWN", "Weekly Score"],
        )

    def test_actionable_selection_applies_rank_liquidity_price_and_foreign_guards(self):
        rankings = pd.DataFrame(
            [
                {
                    "Ticker": "CORE",
                    "Weekly Score": 90.0,
                    "Avg Value/day (B)": 20.0,
                    "Foreign Net10 %Vol": 2.0,
                    "Close": 1_000.0,
                },
                {
                    "Ticker": "THIN",
                    "Weekly Score": 88.0,
                    "Avg Value/day (B)": 5.0,
                    "Foreign Net10 %Vol": 3.0,
                    "Close": 1_000.0,
                },
                {
                    "Ticker": "SELLING",
                    "Weekly Score": 85.0,
                    "Avg Value/day (B)": 30.0,
                    "Foreign Net10 %Vol": -0.1,
                    "Close": 1_000.0,
                },
                {
                    "Ticker": "PENNY",
                    "Weekly Score": 80.0,
                    "Avg Value/day (B)": 30.0,
                    "Foreign Net10 %Vol": 1.0,
                    "Close": 400.0,
                },
                {
                    "Ticker": "CONDITIONAL",
                    "Weekly Score": 70.0,
                    "Avg Value/day (B)": 15.0,
                    "Foreign Net10 %Vol": 0.0,
                    "Close": 800.0,
                },
            ]
        )

        selected = select_actionable_candidates(rankings, candidate_pool=5)

        self.assertEqual(selected["Ticker"].tolist(), ["CORE", "CONDITIONAL"])
        self.assertEqual(selected["Recommendation"].tolist(), ["CORE", "CONDITIONAL"])

    def test_defensive_breakout_rank_rewards_steady_price_action(self):
        steady = make_history(ticker="STEADY", periods=80, start=500, step=1.0)
        volatile = make_history(ticker="VOLATILE", periods=80, start=500, step=1.0)
        oscillation = np.resize(np.array([25.0, -22.0, 18.0, -15.0]), len(volatile))
        volatile["Penutupan"] = volatile["Penutupan"] + oscillation
        volatile["Sebelumnya"] = np.r_[
            volatile.iloc[0]["Penutupan"], volatile["Penutupan"].iloc[:-1]
        ]
        volatile["Open Price"] = volatile["Sebelumnya"] * 1.03
        volatile["Tertinggi"] = volatile[["Open Price", "Penutupan"]].max(axis=1) + 15
        volatile["Terendah"] = volatile[["Open Price", "Penutupan"]].min(axis=1) - 15
        volatile["Volume"] = np.resize(np.array([2_000_000.0, 12_000_000.0]), len(volatile))
        volatile["Nilai"] = volatile["Penutupan"] * volatile["Volume"]
        volatile["Foreign Buy"] = 500_000.0
        volatile["Foreign Sell"] = 1_500_000.0

        ranked = rank_weekly_candidates(
            pd.concat([steady, volatile], ignore_index=True),
            as_of_date=steady["DATE"].max(),
            min_avg_value=100_000_000,
        )

        scores = ranked.set_index("Ticker")["Weekly Score"]
        self.assertGreater(scores["STEADY"], scores["VOLATILE"])
        self.assertTrue(ranked["Latest Active"].all())

    def test_trade_plan_caps_total_exposure_and_uses_next_session_trigger(self):
        histories = [make_history(ticker=f"T{i}", periods=80, start=500 + i * 20) for i in range(5)]
        market = pd.concat(histories, ignore_index=True)
        cutoff = market["DATE"].max()
        ranked = rank_weekly_candidates(
            market, as_of_date=cutoff, min_avg_value=100_000_000
        )
        regime = {
            "label": "RISK-ON",
            "max_exposure_pct": 60.0,
            "risk_cap_pct": 0.75,
        }

        plan = build_weekly_trade_plan(
            market,
            ranked,
            regime,
            as_of_date=cutoff,
            account_size=10_000_000,
            requested_risk_pct=1.0,
            max_positions=3,
        )

        self.assertLessEqual(len(plan), 3)
        self.assertLessEqual(plan["Capital IDR"].sum(), 6_000_000)
        self.assertTrue((plan["Trigger"] == plan["Close Ref"]).all())
        self.assertTrue((plan["Max Entry"] >= plan["Trigger"]).all())
        self.assertTrue((plan["Effective Risk %"] == 0.75).all())
        for column in ["Trigger", "Max Entry", "Stop", "Target 1R", "Target 2R"]:
            for price in plan[column]:
                self.assertEqual(price % idx_tick_size(price), 0)
        self.assertEqual(
            len(plan),
            1,
            "near-duplicate return streams should not create multiple positions",
        )

    def test_existing_holdings_consume_exposure_and_are_not_recommended_again(self):
        histories = [
            make_history(ticker=f"T{i}", periods=80, start=500 + i * 20)
            for i in range(3)
        ]
        market = pd.concat(histories, ignore_index=True)
        cutoff = market["DATE"].max()
        ranked = rank_weekly_candidates(
            market, as_of_date=cutoff, min_avg_value=100_000_000
        )
        regime = {
            "label": "RISK-ON",
            "max_exposure_pct": 25.0,
            "risk_cap_pct": 0.25,
        }

        plan = build_weekly_trade_plan(
            market,
            ranked,
            regime,
            as_of_date=cutoff,
            account_size=10_000_000,
            requested_risk_pct=1.0,
            max_positions=3,
            existing_holdings={"T0": 5_000},
        )

        self.assertTrue(plan.empty, "existing market value already exceeds the 25% cap")


if __name__ == "__main__":
    unittest.main()
