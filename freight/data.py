"""Loading and cleaning of the freight load data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

TARGET = "posted_rate"


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH, parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_PATH, parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(DECEMBER_PATH, parse_dates=["date"])


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city. Every city has a single fixed coordinate in the data."""
    parts = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            part = frame[[role, f"{role}_lat", f"{role}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].median()


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date median of the market-wide signals.

    market_index is essentially one value per day (within-day spread ~0.025 vs
    ~0.17 across days), so the daily median is a good fill for missing values
    and for rows that do not carry the signal at all (the December chart inputs).
    """
    frame = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    return frame.groupby("date")[["market_index", "quote_signal"]].median()


def clean(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues in the input features (never touches the target)."""
    out = frame.copy()

    # Negative weights are sign errors: |w| has the same distribution as positive weights.
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)

    # Missing market_index -> that day's market level.
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    by_date = out["date"].map(market["market_index"])
    out["market_index"] = out["market_index"].fillna(by_date)
    out["market_index"] = out["market_index"].fillna(market["market_index"].median())
    return out


def outlier_mask(rate_per_mile: pd.Series, expected: pd.Series, low: float = 0.7, high: float = 1.4) -> pd.Series:
    """True where the posted rate is implausibly far from the expected rate.

    ~98.5% of loads sit within +-12% of their expected rate; the remaining ~1.5%
    form two clusters at ~0.3x and ~3x (likely unit/entry errors), so a wide
    [0.7, 1.4] band separates them cleanly.
    """
    ratio = rate_per_mile / expected
    return (ratio < low) | (ratio > high) | ~np.isfinite(ratio)
