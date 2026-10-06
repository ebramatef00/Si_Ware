"""Feature engineering.

Features are split in two groups that the model treats differently:
  * structural - what the load is (lane geometry, distance, equipment, weight);
                 learned non-parametrically by gradient boosting.
  * temporal   - when the load moves (market index, position in the quarter,
                 quarter level); modelled with a small linear design so that it
                 can be applied to future dates, which trees cannot extrapolate to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
EARTH_RADIUS_MILES = 3958.8

STRUCTURAL_FEATURES = [
    "distance",
    "log_distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "weight",
    "weight_missing",
    "weight_was_negative",
    "equipment",
]
# Columns that exist for every load but are deliberately NOT model inputs, see README:
#   quote_signal - informative in some months, inverted or pure noise in others;
#                  Nov/Dec match the pure-noise regime.
#   city names   - 8 validation cities never appear in training; coordinates generalise.

# Hinge knots (days before quarter end) for the quarter-end ramp.
RAMP_KNOTS = (3, 7, 14, 21, 28, 42, 56, 70)


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def build_structural(frame: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    x["distance"] = frame["distance"]
    x["log_distance"] = np.log(frame["distance"])
    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        x[col] = frame[col]
    x["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    x["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]
    for col in ("weight", "weight_missing", "weight_was_negative"):
        x[col] = frame[col]
    x["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT)
    selected = features or STRUCTURAL_FEATURES
    for col in selected:  # raw passthrough, used by ablations (e.g. quote_signal)
        if col not in x:
            x[col] = frame[col]
    return x[selected]


def quarter(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("Q")


def days_to_quarter_end(dates: pd.Series) -> np.ndarray:
    end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (end - dates).dt.days.to_numpy()


def ramp_basis(dates: pd.Series, equipment: pd.Series) -> np.ndarray:
    """Equipment-specific hinge basis in days-to-quarter-end.

    Rates climb into every quarter end (+~3% Dry Van, +~4.5% Reefer, +~6% Flatbed
    over the last 8 weeks) and reset when the next quarter starts.
    """
    dtq = days_to_quarter_end(dates)
    hinges = np.column_stack([np.maximum(0, k - dtq) for k in RAMP_KNOTS]).astype(float)
    blocks = [hinges * (equipment.to_numpy() == eq)[:, None] for eq in EQUIPMENT]
    return np.hstack(blocks)
