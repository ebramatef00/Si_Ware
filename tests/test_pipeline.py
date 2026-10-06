import numpy as np
import pandas as pd
import pytest

from freight.data import clean, daily_market, outlier_mask
from freight.features import EQUIPMENT, RAMP_KNOTS, days_to_quarter_end, ramp_basis
from freight.model import PARAMS, FreightRateModel

FAST = dict(PARAMS, max_iter=120, learning_rate=0.1)


def test_days_to_quarter_end():
    dates = pd.Series(pd.to_datetime(["2025-12-31", "2025-10-01", "2025-03-31", "2025-04-01", "2025-11-15"]))
    assert days_to_quarter_end(dates).tolist() == [0, 91, 0, 90, 46]


def test_ramp_basis_is_zero_early_in_quarter_and_split_by_equipment():
    dates = pd.Series(pd.to_datetime(["2025-10-05", "2025-12-30", "2025-12-30"]))
    basis = ramp_basis(dates, pd.Series(["Dry Van", "Dry Van", "Reefer"]))
    k = len(RAMP_KNOTS)
    assert basis.shape == (3, k * len(EQUIPMENT))
    assert np.all(basis[0] == 0)  # 87 days before quarter end: beyond the last knot
    assert basis[1, :k].sum() > 0 and basis[1, k:].sum() == 0  # Dry Van block only
    assert basis[2, 2 * k :].sum() == basis[1, :k].sum()  # same hinge values in the Reefer block


def test_clean_fixes_weight_sign_and_fills_market_from_same_day():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "weight": [-30000.0, np.nan, 20000.0],
        "market_index": [1.0, np.nan, 1.2],
        "quote_signal": [2.0, 2.0, 2.0],
    })
    out = clean(frame, daily_market(frame))
    assert out["weight"].tolist()[0] == 30000.0 and out["weight_was_negative"].tolist() == [1, 0, 0]
    assert out["weight_missing"].tolist() == [0, 1, 0]
    assert out["market_index"].tolist() == [1.0, 1.0, 1.2]


def test_outlier_mask_band():
    mask = outlier_mask(pd.Series([1.0, 0.5, 2.0, 1.3]), pd.Series([1.0, 1.0, 1.0, 1.0]))
    assert mask.tolist() == [False, True, True, False]


def synthetic_loads(n=9000, seed=0):
    """Loads generated from the model's own structure with known parameters."""
    rng = np.random.default_rng(seed)
    cities = pd.DataFrame({"lat": rng.uniform(28, 44, 12), "lon": rng.uniform(-120, -70, 12)})
    pick, drop = rng.integers(0, 12, n), rng.integers(0, 12, n)
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 304, n), unit="D")
    frame = pd.DataFrame({
        "pickup": pick.astype(str), "delivery": drop.astype(str),
        "pickup_lat": cities.lat[pick].to_numpy(), "pickup_lon": cities.lon[pick].to_numpy(),
        "delivery_lat": cities.lat[drop].to_numpy(), "delivery_lon": cities.lon[drop].to_numpy(),
        "distance": rng.uniform(100, 2500, n), "equipment": rng.choice(EQUIPMENT, n),
        "weight": rng.uniform(5000, 45000, n), "date": date,
    })
    day = (frame.date - frame.date.min()).dt.days.to_numpy()
    frame["market_index"] = 1 + 0.12 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 0.02, n)
    frame["quote_signal"] = rng.normal(2, 0.2, n)
    dtq = days_to_quarter_end(frame.date)
    ramp = 0.04 * np.clip((28 - dtq) / 28, 0, None)  # +4% on the last day, starts 4 weeks out
    level = frame.date.dt.quarter.map({1: -0.03, 2: 0.0, 3: 0.02, 4: 0.03}).to_numpy()
    log_rpm = (0.8 - 0.15 * np.log(frame.distance / 500) + 0.1 * (frame.equipment == "Reefer")
               + 0.15 * np.log(frame.market_index) + ramp + level + rng.normal(0, 0.01, n))
    frame["posted_rate"] = np.exp(log_rpm) * frame.distance
    return clean(frame, daily_market(frame))


def test_model_recovers_known_temporal_effects():
    frame = synthetic_loads()
    model = FreightRateModel(params=FAST).fit(frame)
    assert model.coef_[0] == pytest.approx(0.15, abs=0.02)  # market elasticity
    levels = model.quarter_levels()
    assert levels.diff().dropna().tolist() == pytest.approx([0.03, 0.02, 0.01], abs=0.006)
    probe = pd.DataFrame({
        "date": pd.to_datetime(["2025-09-30", "2025-08-01"]), "equipment": ["Dry Van", "Dry Van"],
    })
    k = len(RAMP_KNOTS) * len(EQUIPMENT)
    end, early = ramp_basis(probe.date, probe.equipment) @ model.coef_[1 : 1 + k]
    assert end - early == pytest.approx(0.04, abs=0.008)  # quarter-end premium


def test_predictions_are_positive_and_finite_with_unseen_quarter():
    frame = synthetic_loads(n=4000, seed=1)
    fit, future = frame[frame.date < "2025-10-01"], frame[frame.date >= "2025-10-01"]
    predicted = FreightRateModel(params=FAST).fit(fit).predict(future)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    assert np.median(np.abs(np.log(predicted / future.posted_rate))) < 0.03
