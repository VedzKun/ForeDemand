"""
Conformal Prediction Intervals
--------------------------------
Implements split-conformal (inductive conformal) prediction for time series.

The key idea:
  1. Train the forecaster on a *training* split.
  2. Compute absolute residuals on a *calibration* split (held-out, in-sample).
  3. The (1-α) quantile of those residuals is the conformal width q̂.
  4. For any new forecast ŷ_{t+h}, the prediction interval is:
         [ŷ_{t+h} − q̂_h,  ŷ_{t+h} + q̂_h]

     We compute horizon-specific widths to account for the fact that
     errors grow with forecast horizon.

References
----------
  Angelopoulos & Bates (2023) "Conformal Prediction: A Gentle Introduction"
  Stankeviciute et al. (2021) "Conformal Time-Series Forecasting"
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import Protocol, runtime_checkable


@runtime_checkable
class Forecaster(Protocol):
    """Minimal interface expected from any forecasting model."""

    def fit(self, train: pd.Series) -> "Forecaster": ...
    def predict(self, h: int) -> pd.Series: ...
    def residuals(self) -> pd.Series: ...


class ConformalForecaster:
    """
    Wraps any Forecaster and adds conformal prediction intervals.

    Parameters
    ----------
    model : Forecaster
        A fitted or unfitted model with .fit() / .predict() / .residuals().
    alpha : float
        Miscoverage level.  alpha=0.10 → 90% intervals.
    calib_frac : float
        Fraction of training data used for calibration (held out from the end).
    """

    def __init__(
        self,
        model: Forecaster,
        alpha: float = 0.10,
        calib_frac: float = 0.20,
        fast_calib: bool = False,
    ):
        self.model = model
        self.alpha = alpha
        self.calib_frac = calib_frac
        self.fast_calib = fast_calib  # if True, skip rolling; use in-sample residuals

        # Set after fit
        self._quantile_by_horizon: dict[int, float] = {}
        self._train: pd.Series | None = None
        self._calib: pd.Series | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, series: pd.Series, max_horizon: int = 12) -> "ConformalForecaster":
        """
        Split series into train / calibration, fit model on train,
        then compute conformal widths for horizons 1..max_horizon.
        """
        n = len(series)
        n_calib = max(max_horizon, int(np.ceil(n * self.calib_frac)))
        n_train = n - n_calib

        if n_train < 24:
            raise ValueError(
                f"Training split too small ({n_train} obs). "
                "Reduce calib_frac or provide more data."
            )

        train_s = series.iloc[:n_train]
        calib_s = series.iloc[n_train:]

        self._train = train_s
        self._calib = calib_s
        self._full_series = series  # store for final refit

        # Fit model on training split only
        self.model.fit(train_s)

        if self.fast_calib:
            # Fast path: use in-sample residuals with horizon scaling
            self._fallback_calibration(max_horizon)
            self.model.fit(series)  # refit on full series
        else:
            # Full rolling-origin calibration
            self._calibrate(series, n_train, max_horizon)
        return self

    def _calibrate(
        self, series: pd.Series, n_train: int, max_horizon: int
    ) -> None:
        """
        Rolling-origin calibration to collect horizon-specific absolute errors.
        """
        horizon_errors: dict[int, list[float]] = {h: [] for h in range(1, max_horizon + 1)}

        n = len(series)
        # We need at least one full roll to collect errors
        n_origins = n - n_train
        if n_origins < max_horizon:
            # Not enough points for rolling – fall back to in-sample residuals
            self.model.fit(series.iloc[:n_train])
            self._fallback_calibration(max_horizon)
            return

        for step in range(n_origins - max_horizon + 1):
            origin = n_train + step
            origin_series = series.iloc[:origin]
            # Deep-copy model prototype and reset fitted state
            m = copy.deepcopy(self.model)
            m._train = None
            for attr in ('_result', '_model', '_stl_result', '_ets_result', '_fitted'):
                if hasattr(m, attr):
                    setattr(m, attr, None)
            m.fit(origin_series)
            fc = m.predict(max_horizon)
            for h in range(1, max_horizon + 1):
                if origin + h - 1 < n:
                    actual = series.iloc[origin + h - 1]
                    predicted = fc.iloc[h - 1]
                    horizon_errors[h].append(abs(actual - predicted))

        # Compute (1-α) quantile per horizon
        level = 1.0 - self.alpha
        # Finite-sample correction: use ceil((n_calib+1)(1-α)) / n_calib
        for h in range(1, max_horizon + 1):
            errs = np.array(horizon_errors[h])
            if len(errs) == 0:
                self._quantile_by_horizon[h] = 0.0
            else:
                # Conformalized quantile (Venn–ABERS corrected)
                n_c = len(errs)
                q_idx = int(np.ceil((n_c + 1) * level)) - 1
                q_idx = min(q_idx, n_c - 1)
                self._quantile_by_horizon[h] = float(np.sort(errs)[q_idx])

        # Refit model on the FULL series so predictions start after the last data point
        self.model.fit(self._full_series)

    def _fallback_calibration(self, max_horizon: int) -> None:
        """Use in-sample residuals when calibration data is too thin."""
        res = np.abs(self.model.residuals().values)
        level = 1.0 - self.alpha
        q = float(np.quantile(res, level))
        for h in range(1, max_horizon + 1):
            self._quantile_by_horizon[h] = q * np.sqrt(h)  # grow with horizon

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_with_intervals(self, h: int) -> pd.DataFrame:
        """
        Return a DataFrame with columns: forecast, lower, upper.
        """
        fc = self.model.predict(h)
        lowers, uppers = [], []
        for i in range(1, h + 1):
            q = self._quantile_by_horizon.get(i, self._quantile_by_horizon.get(h, 0.0))
            lowers.append(fc.iloc[i - 1] - q)
            uppers.append(fc.iloc[i - 1] + q)

        result = pd.DataFrame(
            {
                "forecast": fc.values,
                "lower": lowers,
                "upper": uppers,
            },
            index=fc.index,
        )
        result["lower"] = result["lower"].clip(lower=0)  # no negative transactions
        result["model"] = self.model.name
        result["coverage_level"] = 1 - self.alpha
        return result

    @property
    def quantile_widths(self) -> pd.Series:
        """Half-widths (±) at each horizon."""
        return pd.Series(self._quantile_by_horizon, name="conformal_width")


# ── Coverage Checker ─────────────────────────────────────────────────────────

def empirical_coverage(
    actuals: pd.Series,
    intervals: pd.DataFrame,
    lower_col: str = "lower",
    upper_col: str = "upper",
) -> dict:
    """
    Compute empirical coverage and interval metrics.

    Parameters
    ----------
    actuals : pd.Series
    intervals : pd.DataFrame  with lower / upper columns, same index as actuals

    Returns
    -------
    dict with keys: coverage, mean_width, winkler_score
    """
    aligned = intervals.reindex(actuals.index).dropna()
    actual_aligned = actuals.reindex(aligned.index)

    lo = aligned[lower_col].values
    hi = aligned[upper_col].values
    y = actual_aligned.values

    inside = (y >= lo) & (y <= hi)
    coverage = inside.mean()
    mean_width = (hi - lo).mean()

    # Winkler score (lower = better; penalises misses more than width)
    alpha = 1 - intervals["coverage_level"].iloc[0] if "coverage_level" in intervals.columns else 0.10
    winkler = _winkler(y, lo, hi, alpha)

    return {
        "coverage": coverage,
        "target_coverage": 1 - alpha,
        "mean_width": mean_width,
        "winkler_score": winkler,
        "n_obs": len(y),
    }


def _winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    width = hi - lo
    penalty_lo = (2 / alpha) * (lo - y) * (y < lo)
    penalty_hi = (2 / alpha) * (y - hi) * (y > hi)
    return float((width + penalty_lo + penalty_hi).mean())
