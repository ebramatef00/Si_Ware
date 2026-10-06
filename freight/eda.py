"""Exploratory findings used in the report: data quality, quote_signal regimes,
the quarter-end cycle. Every number in the report is computed here."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd

from .data import TARGET
from .features import EQUIPMENT, RAMP_KNOTS, ramp_basis

INK, ACCENT, MUTED, BAD = "#064A56", "#D1495B", "#9DAFB3", "#EDAE49"


def data_quality(train_raw: pd.DataFrame, valid_raw: pd.DataFrame, n_outliers: int) -> pd.DataFrame:
    """Issues found, with counts in train / validation and how each was handled."""
    train_cities = set(train_raw["pickup"]) | set(train_raw["delivery"])
    unseen = ~valid_raw["pickup"].isin(train_cities) | ~valid_raw["delivery"].isin(train_cities)
    rows = [
        ("Negative weights", (train_raw["weight"] < 0).sum(), (valid_raw["weight"] < 0).sum(),
         "They look like sign errors (flipped, they match the normal weights), so I flip them and keep a flag"),
        ("Missing weights", train_raw["weight"].isna().sum(), valid_raw["weight"].isna().sum(),
         "Left as missing with a flag; the tree model handles gaps on its own"),
        ("Missing market index", train_raw["market_index"].isna().sum(), valid_raw["market_index"].isna().sum(),
         "Filled with that day's median, since the index is almost the same for every load on a given day"),
        ("Prices that are clearly wrong (a fifth to five times the expected rate)", n_outliers, "n/a",
         "Left out of training. Their distances look normal, so it's the price that's wrong"),
        ("Loads involving a city that's not in training", 0, int(unseen.sum()),
         "I don't use city names, only coordinates, so new cities still get sensible predictions"),
        ("Coordinates cut off at the edge of a box (Laredo, Boston, Providence)", "-", "-",
         "Kept as they are; each city still has one consistent location"),
        ("Distances that bottom out at 70 miles", int((train_raw["distance"] <= 70).sum()), int((valid_raw["distance"] <= 70).sum()),
         "Kept, but I avoided straight-line distance features, which go haywire on these short lanes"),
        ("quote_signal means different things in different months", "10 months", "2 months",
         "Left out, because November and December look like the months where it's pure noise"),
    ]
    return pd.DataFrame(rows, columns=["Problem", "Train", "Validation", "What I did"])


def quote_signal_regimes(train_raw: pd.DataFrame, valid_raw: pd.DataFrame) -> pd.DataFrame:
    """Per month: rank correlation of quote_signal with log distance (label-free), and
    for labelled months the share of loads whose quote is within 5% of rate/mile.
    Real rate/mile falls with distance (rho ~ -0.74), so an informative quote must too."""
    rows = []
    for frame, labelled in ((train_raw, True), (valid_raw, False)):
        for month, g in frame.groupby(frame["date"].dt.month):
            rho = g["quote_signal"].corr(np.log(g["distance"]), method="spearman")
            match = ((g["quote_signal"] / (g[TARGET] / g["distance"]) - 1).abs() < 0.05).mean() if labelled else np.nan
            rows.append({"month": month, "rho_quote_vs_log_distance": rho, "share_quote_within_5pct": match})
    out = pd.DataFrame(rows)
    out["regime"] = np.select(
        [out["rho_quote_vs_log_distance"] < -0.5, out["rho_quote_vs_log_distance"] > 0.5], ["informative", "inverted"], "noise"
    )
    return out


def fig_quote_regimes(regimes: pd.DataFrame, path) -> None:
    palette = {"informative": INK, "inverted": BAD, "noise": ACCENT}
    labels = pd.to_datetime(regimes["month"].astype(str) + "-2025", format="%m-%Y").dt.strftime("%b")
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=160)
    ax.bar(labels, regimes["rho_quote_vs_log_distance"], color=regimes["regime"].map(palette))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvspan(9.5, 11.5, color=MUTED, alpha=0.15)
    ax.text(10.5, 0.55, "validation\n(Nov-Dec)", ha="center", fontsize=8, color="#455A60")
    for x, (rho, regime) in enumerate(zip(regimes["rho_quote_vs_log_distance"], regimes["regime"])):
        if regime == "noise":
            ax.text(x, rho + 0.04, f"{rho:+.2f}", ha="center", fontsize=7, color=ACCENT)
    ax.set_ylabel("Spearman(quote, log distance)")
    ax.set_title("quote_signal changes meaning by month", loc="left", fontsize=10, fontweight="bold")
    handles = [Patch(color=color, label=regime) for regime, color in palette.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def time_effect(model, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Daily price level after removing what the load is (structural GBM) and the
    market index, next to the model's fitted ramp + quarter level. Train days show
    the observed effect; Nov-Dec show the forecast."""
    comp = model.components(train)
    observed = np.log(train[TARGET] / train["distance"]) - comp["structural"] - comp["market"]
    ok = (observed - comp["ramp"] - comp["level"]).abs() < np.log(1.4)  # drop mislabelled rates
    daily = pd.DataFrame({"date": train["date"], "observed": observed, "fitted": comp["ramp"] + comp["level"]})[ok]
    daily = daily.groupby("date").mean()
    vcomp = model.components(valid)
    future = pd.DataFrame({"date": valid["date"], "fitted": vcomp["ramp"] + vcomp["level"]}).groupby("date").mean()
    return pd.concat([daily, future]).sort_index()


def fig_quarter_cycle(effect: pd.DataFrame, path) -> None:
    centre = effect["fitted"].mean()
    pct = lambda s: 100 * (np.exp(s - centre) - 1)
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=160)
    ax.scatter(effect.index, pct(effect["observed"]), s=6, color=MUTED, label="observed daily level (train)")
    ax.plot(effect.index, pct(effect["fitted"]), color=INK, lw=2, label="model: quarter level + quarter-end ramp")
    for start in pd.date_range("2025-04-01", "2025-10-01", freq="QS"):
        ax.axvline(start, color=ACCENT, lw=0.8, ls="--")
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=MUTED, alpha=0.15)
    ax.text(pd.Timestamp("2025-12-01"), ax.get_ylim()[1] * 0.85, "forecast", ha="center", fontsize=8, color="#455A60")
    ax.set_ylabel("price level vs. average (%)")
    ax.set_title("Rates climb into every quarter end, then reset (market index removed)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ramp_by_equipment(model, days=(56, 28, 14, 7, 0)) -> pd.DataFrame:
    """Fitted quarter-end premium (%) by equipment at a few days before quarter end."""
    k = 1 if model.use_market else 0
    coef = model.coef_[k : k + len(EQUIPMENT) * len(RAMP_KNOTS)]
    dates = pd.Series(pd.Timestamp("2025-12-31") - pd.to_timedelta(list(days), unit="D"))
    rows = {}
    for eq in EQUIPMENT:
        ramp = ramp_basis(dates, pd.Series([eq] * len(dates))) @ coef
        rows[eq] = np.round(100 * (np.exp(ramp) - 1), 2)
    return pd.DataFrame(rows, index=["on the last day" if d == 0 else f"{d // 7} week{'s' if d >= 14 else ''} before" for d in days]).T


def fig_validation(summary: pd.DataFrame, path) -> None:
    level = summary.index.get_level_values
    s = summary[(level("family") == "quarter") & (level("subset") == "clean")].droplevel(["family", "subset"])["MAPE_%"].sort_values()
    colors = [ACCENT if "FINAL" in m else (MUTED if m.startswith("baseline") else INK) for m in s.index]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.barh(s.index[::-1], s.values[::-1], color=colors[::-1])
    for i, v in enumerate(s.values[::-1]):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("MAPE on held-out months 2-3 of Q2 and Q3 (clean labels)")
    ax.set_title("Backtest shaped like the real task", loc="left", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
