"""
Forecasting Models
-------------------
Implements three models used in ForeDemand:

1. SeasonalNaive   – Last-year same-month carry-forward (baseline)
2. ETSModel        – Exponential Smoothing (additive trend + multiplicative seasonality on log)
3. HybridForecaster – STL decomposition + ETS on remainder (main model)

All models share a common interface:
    .fit(train: pd.Series) -> self
    .predict(h: int) -> pd.Series          (point forecasts, original scale)
    .residuals() -> pd.Series              (in-sample, original scale)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")


# ── helpers ──────────────────────────────────────────────────────────────────

def _future_index(last_date: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + pd.DateOffset(months=1), periods=h, freq="MS")


# ── 1. Seasonal Naive ────────────────────────────────────────────────────────

class SeasonalNaive:
    """Carry-forward the value from the same month last year (s=12)."""

    name = "Seasonal Naive"

    def __init__(self, season: int = 12):
        self.season = season
        self._train: pd.Series | None = None

    def fit(self, train: pd.Series) -> "SeasonalNaive":
        self._train = train.copy()
        return self

    def predict(self, h: int) -> pd.Series:
        if self._train is None:
            raise RuntimeError("Call fit() first.")
        preds = []
        extended = list(self._train.values)
        for _ in range(h):
            preds.append(extended[-self.season])
            extended.append(extended[-self.season])
        idx = _future_index(self._train.index[-1], h)
        return pd.Series(preds, index=idx, name=self.name)

    def residuals(self) -> pd.Series:
        """In-sample fitted values subtracted from actuals."""
        train = self._train
        fitted = train.shift(self.season)
        res = train - fitted
        return res.dropna()


# ── 2. ETS (Exponential Smoothing) ──────────────────────────────────────────

class ETSModel:
    """
    Holt-Winters multiplicative seasonality on the raw series.
    Falls back to additive if MLE fails.
    """

    name = "ETS"

    def __init__(self, season: int = 12):
        self.season = season
        self._model = None
        self._result = None
        self._train: pd.Series | None = None

    def fit(self, train: pd.Series) -> "ETSModel":
        self._train = train.copy()
        try:
            self._model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="mul",
                seasonal_periods=self.season,
                initialization_method="estimated",
            )
            self._result = self._model.fit(optimized=True, remove_bias=True)
        except Exception:
            # Fallback: additive seasonality
            self._model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=self.season,
                initialization_method="estimated",
            )
            self._result = self._model.fit(optimized=True)
        return self

    def predict(self, h: int) -> pd.Series:
        fc = self._result.forecast(h)
        fc.index = _future_index(self._train.index[-1], h)
        fc.name = self.name
        return fc

    def residuals(self) -> pd.Series:
        fitted = self._result.fittedvalues
        res = self._train - fitted
        res.name = self.name
        return res


# ── 3. Hybrid: STL + ETS on remainder ───────────────────────────────────────

class HybridForecaster:
    """
    STL decomposition on log-scale series.
    ETS fitted on the seasonally-adjusted (trend+remainder) component.
    Forecast = exp(ETS_forecast + seasonal_factor).
    """

    name = "STL-ETS (Hybrid)"

    def __init__(self, season: int = 12, stl_robust: bool = True):
        self.season = season
        self.stl_robust = stl_robust
        self._train: pd.Series | None = None
        self._log_train: pd.Series | None = None
        self._stl_result = None
        self._ets_result = None
        self._seasonal_factors: np.ndarray | None = None
        self._fitted: pd.Series | None = None

    def fit(self, train: pd.Series) -> "HybridForecaster":
        self._train = train.copy()
        log_s = np.log(train)
        self._log_train = log_s

        # STL decomposition
        stl = STL(log_s, period=self.season, robust=self.stl_robust)
        self._stl_result = stl.fit()

        # Seasonally-adjusted = trend + remainder
        sa = self._stl_result.trend + self._stl_result.resid

        # Fit ETS on seasonally-adjusted log series
        ets_model = ExponentialSmoothing(
            sa,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        self._ets_result = ets_model.fit(optimized=True)

        # Store the last full seasonal cycle (12 months)
        self._seasonal_factors = self._stl_result.seasonal[-self.season:]

        # In-sample fitted (log scale)
        log_fitted = self._ets_result.fittedvalues + self._stl_result.seasonal
        self._fitted = np.exp(log_fitted)
        self._fitted.name = self.name
        return self

    def predict(self, h: int) -> pd.Series:
        # ETS forecast on SA series
        sa_fc = self._ets_result.forecast(h)
        # Repeat seasonal factors cyclically
        n_cycles = (h // self.season) + 2
        repeated = np.tile(self._seasonal_factors, n_cycles)[:h]
        log_fc = sa_fc.values + repeated
        fc = np.exp(log_fc)
        idx = _future_index(self._train.index[-1], h)
        return pd.Series(fc, index=idx, name=self.name)

    def residuals(self) -> pd.Series:
        res = self._train - self._fitted
        res.name = self.name
        return res


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.upi_data import load_upi_data

    df = load_upi_data()
    series = df["transactions_mn"]
    train = series[:-12]

    for Model in [SeasonalNaive, ETSModel, HybridForecaster]:
        m = Model().fit(train)
        fc = m.predict(12)
        print(f"\n{m.name}")
        print(fc)
