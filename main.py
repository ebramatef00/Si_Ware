"""Freight rate prediction pipeline.

    python main.py validate   # backtests + baselines + ablations -> reports/
    python main.py predict    # train on all data, write both prediction files
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from freight import data, validate
from freight.model import FreightRateModel

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"


def load_clean():
    train, valid = data.load_train(), data.load_validation()
    # Market signals are features (not the target), so pooling train + validation
    # rows to estimate each day's market level leaks no label information.
    market = data.daily_market(train, valid)
    return data.clean(train, market), data.clean(valid, market), market, data.city_coordinates(train, valid)


def december_frame(market: pd.DataFrame, coords: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The December chart inputs carry no coordinates and no market_index: take the
    coordinates from the city table and each day's market index from the
    validation loads moving on that same day (feature data, no labels)."""
    raw = pd.read_csv(data.DECEMBER_PATH)
    frame = raw.drop(columns="predicted_rate").assign(date=pd.to_datetime(raw["date"]))
    for role in ("pickup", "delivery"):
        frame[f"{role}_lat"] = frame[role].map(coords["lat"])
        frame[f"{role}_lon"] = frame[role].map(coords["lon"])
    frame["market_index"] = frame["date"].map(market["market_index"])
    if frame[["pickup_lat", "delivery_lat", "market_index"]].isna().any().any():
        raise SystemExit("December inputs could not be fully resolved (unknown city or date)")
    return raw, data.clean(frame, market)


def cmd_validate(_args) -> None:
    train, *_ = load_clean()
    print(f"Running validation on {len(train):,} labelled loads ...")
    results = validate.run(train)
    summary = validate.summarise(results)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "validation_results.csv", index=False)
    summary.to_csv(REPORTS / "validation_summary.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary)


def cmd_predict(_args) -> None:
    train, valid, market, coords = load_clean()
    model = FreightRateModel().fit(train)
    print(f"Trained on {len(train):,} loads ({model.n_outliers_} mislabelled rates excluded)")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")

    template = pd.read_csv(data.TEMPLATE_PATH)
    predicted = pd.Series(model.predict(valid), index=valid["load_id"])
    template["predicted_rate"] = template["load_id"].map(predicted).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template[["load_id", "predicted_rate"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(template):,} predictions -> {PREDICTIONS_PATH.name}")

    raw, december = december_frame(market, coords)
    raw["predicted_rate"] = np.round(model.predict(december), 2)
    raw.to_csv(data.DECEMBER_PATH, index=False)
    print(f"Filled December chart inputs -> {data.DECEMBER_PATH.relative_to(ROOT)}")
    print(raw[["date", "predicted_rate"]].describe().loc[["min", "mean", "max"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("predict").set_defaults(func=cmd_predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
