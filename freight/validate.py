"""Validation harness.

The final task is: train on Jan-Oct 2025, predict Nov-Dec 2025.
  * Nov-Dec are months 2-3 of Q4, and October (Q4 month 1) is in training.
  * 12% of the loads touch one of 8 cities that never appear in training.

The folds reproduce both properties, so the backtest measures the same problem:
  quarter    train on everything before month 2 of a quarter, test months 2-3
             (Q2: Jan-Apr -> May-Jun, Q3: Jan-Jul -> Aug-Sep)
  new_city   the same folds with 8 random cities removed from training,
             scored only on test loads that touch a removed city
  next_qtr   stress test: train Jan-Sep -> test Oct (quarter level unseen)
Rows are never shuffled across time: a random split would leak each day's
market level and position in the quarter into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import TARGET
from .features import STRUCTURAL_FEATURES, build_structural
from .model import PARAMS, FreightRateModel

QUARTER_FOLDS = {"Q2": ("2025-05-01", "2025-07-01"), "Q3": ("2025-08-01", "2025-10-01")}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    ape = np.abs(err) / actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * np.mean(ape)),
        "MedAPE_%": float(100 * np.median(ape)),
        "bias_%": float(100 * np.median(np.log(predicted / actual))),
    }


def reference_outliers(train: pd.DataFrame) -> pd.Series:
    """Mislabelled rates, flagged once on all labelled data. Used only to report
    'clean' metrics, so the model under test does not grade its own outliers."""
    model = FreightRateModel()
    y = np.log(train[TARGET] / train["distance"]).to_numpy()
    return pd.Series(model._flag_outliers(train, build_structural(train), y), index=train.index)


# ---------------------------------------------------------------- baselines
class GlobalRatePerMile:
    def fit(self, train):
        self.rpm_ = (train[TARGET] / train["distance"]).median()
        return self

    def predict(self, frame):
        return self.rpm_ * frame["distance"].to_numpy()


class EquipmentDistanceBand:
    """Median rate/mile per (equipment, distance decile)."""

    def fit(self, train):
        self.edges_ = np.quantile(train["distance"], np.linspace(0, 1, 11))[1:-1]
        self.table_ = (train[TARGET] / train["distance"]).groupby(self._key(train)).median()
        return self

    def _key(self, frame):
        band = pd.Series(np.digitize(frame["distance"], self.edges_), index=frame.index).astype(str)
        return frame["equipment"] + "|" + band

    def predict(self, frame):
        return self._key(frame).map(self.table_).to_numpy() * frame["distance"].to_numpy()


class LaneMedian:
    """Median rate/mile per (pickup, delivery, equipment); falls back to the band baseline."""

    def fit(self, train):
        self.fallback_ = EquipmentDistanceBand().fit(train)
        rpm = train[TARGET] / train["distance"]
        self.table_ = rpm.groupby([train["pickup"], train["delivery"], train["equipment"]]).median()
        return self

    def predict(self, frame):
        idx = pd.MultiIndex.from_frame(frame[["pickup", "delivery", "equipment"]])
        rpm = pd.Series(self.table_.reindex(idx).to_numpy(), index=frame.index)
        fallback = pd.Series(self.fallback_.predict(frame) / frame["distance"].to_numpy(), index=frame.index)
        return rpm.fillna(fallback).to_numpy() * frame["distance"].to_numpy()


class NaiveGBM:
    """The 'obvious' approach: one gradient-boosted model on every column."""

    COLUMNS = STRUCTURAL_FEATURES + ["market_index", "quote_signal", "day_of_week", "month", "day"]

    def _x(self, frame):
        f = frame.assign(day_of_week=frame["date"].dt.dayofweek, month=frame["date"].dt.month, day=frame["date"].dt.day)
        return build_structural(f, self.COLUMNS)

    def fit(self, train):
        y = np.log(train[TARGET] / train["distance"])
        self.gbm_ = HistGradientBoostingRegressor(**PARAMS).fit(self._x(train), y)
        return self

    def predict(self, frame):
        return np.exp(self.gbm_.predict(self._x(frame))) * frame["distance"].to_numpy()


CANDIDATES = {
    "baseline: global $/mile": GlobalRatePerMile,
    "baseline: equipment x distance band": EquipmentDistanceBand,
    "baseline: lane median": LaneMedian,
    "naive GBM on all columns": NaiveGBM,
    "FINAL model": FreightRateModel,
    "final - no outlier removal": lambda: FreightRateModel(remove_outliers=False),
    "final - no market index": lambda: FreightRateModel(use_market=False),
    "final - no quarter-end ramp": lambda: FreightRateModel(use_ramp=False),
    "final - no quarter level": lambda: FreightRateModel(use_quarter_level=False),
    "final + quote_signal": lambda: FreightRateModel(structural=STRUCTURAL_FEATURES + ["quote_signal"]),
}
# Candidates also scored on the unseen-city and next-quarter folds.
CORE = ["baseline: equipment x distance band", "baseline: lane median", "naive GBM on all columns", "FINAL model"]


# ---------------------------------------------------------------- folds
def folds(train: pd.DataFrame, n_city_draws: int = 2, n_cities: int = 8):
    date = train["date"]
    rng = np.random.default_rng(0)
    cities = np.array(sorted(set(train["pickup"]) | set(train["delivery"])))
    for name, (start, end) in QUARTER_FOLDS.items():
        fit_mask, test_mask = date < start, (date >= start) & (date < end)
        yield "quarter", f"{name}: <{start} -> [{start}, {end})", fit_mask, test_mask, None
        for draw in range(n_city_draws):
            held = {str(c) for c in rng.choice(cities, size=n_cities, replace=False)}
            touches = train["pickup"].isin(held) | train["delivery"].isin(held)
            yield "new_city", f"{name} draw {draw}: {sorted(held)}", fit_mask & ~touches, test_mask & touches, CORE
    yield "next_qtr", "<2025-10-01 -> October", date < "2025-10-01", date >= "2025-10-01", CORE


def run(train: pd.DataFrame, candidates: dict | None = None) -> pd.DataFrame:
    candidates = candidates or CANDIDATES
    is_outlier = reference_outliers(train)
    rows = []
    for family, fold, fit_mask, test_mask, only in folds(train):
        fit_frame, test_frame = train[fit_mask], train[test_mask]
        actual = test_frame[TARGET].to_numpy()
        clean = ~is_outlier[test_mask].to_numpy()
        month = test_frame["date"].dt.month.to_numpy()
        for name, factory in candidates.items():
            if only is not None and name not in only:
                continue
            predicted = factory().fit(fit_frame).predict(test_frame)
            for subset, mask in (("clean", clean), ("all", np.ones_like(clean))):
                rows.append({"family": family, "fold": fold, "model": name, "subset": subset, "month": "all",
                             "n": int(mask.sum()), **metrics(actual[mask], predicted[mask])})
                for m in np.unique(month):
                    sel = mask & (month == m)
                    rows.append({"family": family, "fold": fold, "model": name, "subset": subset, "month": int(m),
                                 "n": int(sel.sum()), **metrics(actual[sel], predicted[sel])})
            last = [r for r in rows if r["model"] == name and r["fold"] == fold and r["subset"] == "clean" and r["month"] == "all"][-1]
            print(f"  {family:9s} {fold[:24]:24s} {name:38s} MAPE(clean) {last['MAPE_%']:.2f}%  bias {last['bias_%']:+.2f}%", flush=True)
    return pd.DataFrame(rows)


def random_split_illusion(train: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Score the naive GBM on a shuffled 80/20 split and on the Q3-aligned fold.
    The gap is how much a random split would have overstated accuracy."""
    is_outlier = reference_outliers(train)
    rng = np.random.default_rng(seed)
    shuffled = rng.random(len(train)) < 0.8
    start, end = QUARTER_FOLDS["Q3"]
    q3_fit, q3_test = train["date"] < start, (train["date"] >= start) & (train["date"] < end)
    rows = []
    for split, fit_mask, test_mask in (("random 80/20", shuffled, ~shuffled), ("Q3-aligned (time)", q3_fit, q3_test)):
        test = train[test_mask]
        clean = ~is_outlier[test_mask].to_numpy()
        predicted = NaiveGBM().fit(train[fit_mask]).predict(test)
        rows.append({"split": split, **metrics(test[TARGET].to_numpy()[clean], predicted[clean])})
    return pd.DataFrame(rows).set_index("split").round(2)


def pick(summary: pd.DataFrame, family: str, subset: str) -> pd.DataFrame:
    """Rows of `summarise` output for one fold family and label subset, indexed by model."""
    level = summary.index.get_level_values
    return summary[(level("family") == family) & (level("subset") == subset)].droplevel(["family", "subset"])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds within each family, all test months pooled."""
    pooled = results[results["month"] == "all"]
    return (
        pooled.groupby(["family", "subset", "model"], sort=False)[["MAE", "RMSE", "MAPE_%", "MedAPE_%", "bias_%"]]
        .mean()
        .round(2)
    )
