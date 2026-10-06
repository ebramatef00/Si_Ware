"""Semi-parametric freight rate model.

    log(rate / mile) = GBM(structural features)                 what the load is
                     + beta * log(market_index)                  daily market
                     + ramp_equipment(days to quarter end)       quarter-end crunch
                     + level(quarter)                            price level
                     + noise

Why not a single gradient-boosted model on every column?  Trees cannot
extrapolate in time.  market_index in the forecast months sits at January
levels while prices have drifted ~6% higher since, so a tree maps "low index"
to January prices and under-predicts.  Estimating the market elasticity *within*
quarters and anchoring the level on the latest quarter removes that confusion.

The two parts are fitted by backfitting: GBM on (y - linear part), then least
squares on (y - GBM), repeated until stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import TARGET, outlier_mask
from .features import build_structural, days_to_quarter_end, quarter, ramp_basis

PARAMS = dict(
    learning_rate=0.05,
    max_iter=800,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    categorical_features="from_dtype",
    early_stopping=False,
    random_state=0,
)
# The outlier screen only needs a rough expected rate for a wide [0.7, 1.4] band.
SCREEN_PARAMS = dict(PARAMS, learning_rate=0.08, max_iter=300)


@dataclass
class FreightRateModel:
    use_market: bool = True
    use_ramp: bool = True
    use_quarter_level: bool = True
    remove_outliers: bool = True
    structural: list[str] | None = None
    backfit_iters: int = 3
    params: dict = field(default_factory=lambda: dict(PARAMS))

    # ------------------------------------------------------------------ design
    def _temporal(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        if self.use_market:
            cols.append(np.log(frame["market_index"].to_numpy())[:, None])
        if self.use_ramp:
            cols.append(ramp_basis(frame["date"], frame["equipment"]))
        if self.use_quarter_level:
            q = quarter(frame["date"])
            cols.append(np.column_stack([(q == p).to_numpy(float) for p in self.quarters_]))
        return np.hstack(cols) if cols else np.zeros((len(frame), 0))

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "FreightRateModel":
        y = np.log(train[TARGET] / train["distance"]).to_numpy()
        x = build_structural(train, self.structural)

        keep = np.ones(len(train), dtype=bool)
        if self.remove_outliers:
            keep = ~self._flag_outliers(train, x, y)
        self.n_outliers_ = int((~keep).sum())
        train, x, y = train[keep], x[keep], y[keep]

        self.quarters_ = sorted(quarter(train["date"]).unique())
        t = self._temporal(train)
        linear = np.zeros(len(y))
        self.coef_ = np.zeros(t.shape[1])
        for _ in range(self.backfit_iters):
            self.gbm_ = HistGradientBoostingRegressor(**self.params).fit(x, y - linear)
            if t.shape[1] == 0:
                break
            self.coef_, *_ = np.linalg.lstsq(t, y - self.gbm_.predict(x), rcond=None)
            linear = t @ self.coef_
        return self

    def _flag_outliers(self, train: pd.DataFrame, x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Robust in-sample fit (absolute error) that also sees time, used only to
        flag implausible labels; never used for prediction."""
        xr = x.assign(
            market_index=train["market_index"].to_numpy(),
            days_to_quarter_end=days_to_quarter_end(train["date"]),
            month=train["date"].dt.month.to_numpy(),
        )
        robust = HistGradientBoostingRegressor(loss="absolute_error", **SCREEN_PARAMS).fit(xr, y)
        return outlier_mask(pd.Series(np.exp(y)), pd.Series(np.exp(robust.predict(xr)))).to_numpy()

    # ------------------------------------------------------------------ predict
    def quarter_levels(self) -> pd.Series:
        n = len(self.quarters_)
        return pd.Series(self.coef_[-n:], index=self.quarters_) if self.use_quarter_level else pd.Series(dtype=float)

    def _level(self, frame: pd.DataFrame) -> np.ndarray:
        """Quarter level: fitted value for quarters seen in training; for later
        quarters, the last level plus the average quarter-on-quarter drift."""
        levels = self.quarter_levels()
        drift = (levels.iloc[-1] - levels.iloc[0]) / (len(levels) - 1) if len(levels) > 1 else 0.0
        last = levels.index[-1]
        q = quarter(frame["date"])
        return np.array([levels[p] if p in levels.index else levels.iloc[-1] + drift * (p - last).n for p in q])

    def components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Additive contributions to log(rate per mile), one column per model part."""
        out = pd.DataFrame(index=frame.index)
        out["structural"] = self.gbm_.predict(build_structural(frame, self.structural))
        out["market"], out["ramp"], out["level"] = 0.0, 0.0, 0.0
        k = 0
        if self.use_market:
            out["market"] = self.coef_[0] * np.log(frame["market_index"].to_numpy())
            k = 1
        if self.use_ramp:
            ramp = ramp_basis(frame["date"], frame["equipment"])
            out["ramp"] = ramp @ self.coef_[k : k + ramp.shape[1]]
        if self.use_quarter_level:
            out["level"] = self._level(frame)
        return out

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.components(frame).sum(axis=1).to_numpy()) * frame["distance"].to_numpy()
