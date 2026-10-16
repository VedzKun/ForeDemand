"""
Backtesting Engine
-------------------
Implements rolling-origin (expanding-window) backtest for comparing
multiple models on point accuracy and interval coverage.

Metrics computed:
  - MAE  : Mean Absolute Error
  - RMSE : Root Mean Squared Error
  - MASE : Mean Absolute Scaled Error (vs. seasonal naive)
  - sMAPE: Symmetric MAPE
  - Coverage : Empirical coverage of conformal intervals
  - Winkler  : Interval sharpness score (lower = better)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    origin: pd.Timestamp
    horizon: int
    model_name: str
    actuals: pd.Series
    forecasts: pd.Series
    lower: pd.Series
    upper: pd.Series
    coverage_level: float

    @property
    def errors(self) -> np.ndarray:
        return (self.actuals - self.forecasts).values

    @property
    def abs_errors(self) -> np.ndarray:
        return np.abs(self.errors)


@dataclass
class BacktestSummary:
    model_name: str
    mae: float
    rmse: float
    smape: float
    mase: float
    coverage: float
    target_coverage: float
    mean_width: float
    winkler_score: float
    n_forecasts: int
    fold_results: List[FoldResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "Model": self.model_name,
            "MAE": round(self.mae, 1),
            "RMSE": round(self.rmse, 1),
            "sMAPE (%)": round(self.smape, 2),
            "MASE": round(self.mase, 3),
            "Coverage": f"{self.coverage*100:.1f}%",
            "Target": f"{self.target_coverage*100:.0f}%",
            "d.Coverage": f"{(self.coverage - self.target_coverage)*100:+.1f}%",
            "Mean Width": round(self.mean_width, 1),
            "Winkler ↓": round(self.winkler_score, 1),
        }


# ── Backtest Engine ───────────────────────────────────────────────────────────

class Backtester:
    """
    Rolling-origin backtester with conformal interval evaluation.

    Parameters
    ----------
    series : pd.Series   Full time series (transactions_mn)
    min_train : int      Minimum training window (months)
    horizon : int        Forecast horizon per origin (months)
    step : int           Step size between origins (1 = every month)
    alpha : float        Conformal miscoverage level (0.10 → 90% intervals)
    calib_frac : float   Fraction of training data used for calibration
    """

    def __init__(
        self,
        series: pd.Series,
        min_train: int = 36,
        horizon: int = 6,
        step: int = 3,
        alpha: float = 0.10,
        calib_frac: float = 0.25,
    ):
        self.series = series
        self.min_train = min_train
        self.horizon = horizon
        self.step = step
        self.alpha = alpha
        self.calib_frac = calib_frac

        self._results: dict[str, List[FoldResult]] = {}

    def run(self, models: list) -> "Backtester":
        """
        Run backtest for each model and store results.

        Parameters
        ----------
        models : list of instantiated (unfitted) Forecaster objects
        """
        from models.conformal import ConformalForecaster
        import copy

        n = len(self.series)
        origins = list(range(self.min_train, n - self.horizon + 1, self.step))

        for model_proto in models:
            model_name = model_proto.name
            fold_results: List[FoldResult] = []

            for origin_idx in origins:
                train_s = self.series.iloc[:origin_idx]
                actual_s = self.series.iloc[origin_idx: origin_idx + self.horizon]

                # Deep-copy the prototype to get a fresh unfitted instance
                model_copy = copy.deepcopy(model_proto)
                # Reset any fitted state
                model_copy._train = None
                if hasattr(model_copy, '_result'):    model_copy._result = None
                if hasattr(model_copy, '_model'):     model_copy._model = None
                if hasattr(model_copy, '_stl_result'): model_copy._stl_result = None
                if hasattr(model_copy, '_ets_result'): model_copy._ets_result = None
                if hasattr(model_copy, '_fitted'):    model_copy._fitted = None

                try:
                    cf = ConformalForecaster(
                        model_copy,
                        alpha=self.alpha,
                        calib_frac=self.calib_frac,
                        fast_calib=True,  # use in-sample residuals for speed
                    )
                    cf.fit(train_s, max_horizon=self.horizon)
                    pred_df = cf.predict_with_intervals(self.horizon)
                except Exception as e:
                    warnings.warn(f"[{model_name}] origin={origin_idx}: {e}")
                    continue

                # Align predictions to actuals
                common_idx = actual_s.index.intersection(pred_df.index)
                if len(common_idx) == 0:
                    continue

                fold_results.append(
                    FoldResult(
                        origin=self.series.index[origin_idx],
                        horizon=self.horizon,
                        model_name=model_name,
                        actuals=actual_s.reindex(common_idx),
                        forecasts=pred_df.loc[common_idx, "forecast"],
                        lower=pred_df.loc[common_idx, "lower"],
                        upper=pred_df.loc[common_idx, "upper"],
                        coverage_level=1 - self.alpha,
                    )
                )

            self._results[model_name] = fold_results
            print(f"      [{model_name}] {len(fold_results)} folds completed")

        return self

    # ── Metric computation ────────────────────────────────────────────────────

    def _seasonal_naive_mae(self) -> float:
        """In-sample MAE of seasonal naive for MASE denominator."""
        s = self.series
        shifted = s.shift(12)
        return float(np.nanmean(np.abs((s - shifted).values)))

    def summarise(self) -> pd.DataFrame:
        """Return a DataFrame of metric summaries for all models."""
        naive_mae = self._seasonal_naive_mae()
        rows = []
        for model_name, folds in self._results.items():
            if not folds:
                continue
            all_y = np.concatenate([f.actuals.values for f in folds])
            all_yhat = np.concatenate([f.forecasts.values for f in folds])
            all_lo = np.concatenate([f.lower.values for f in folds])
            all_hi = np.concatenate([f.upper.values for f in folds])

            mae = float(np.mean(np.abs(all_y - all_yhat)))
            rmse = float(np.sqrt(np.mean((all_y - all_yhat) ** 2)))
            smape = float(
                100 * np.mean(
                    2 * np.abs(all_y - all_yhat) / (np.abs(all_y) + np.abs(all_yhat) + 1e-9)
                )
            )
            mase = mae / (naive_mae + 1e-9)

            inside = (all_y >= all_lo) & (all_y <= all_hi)
            coverage = float(inside.mean())
            mean_width = float((all_hi - all_lo).mean())
            winkler = float(
                (
                    (all_hi - all_lo)
                    + (2 / self.alpha) * (all_lo - all_y) * (all_y < all_lo)
                    + (2 / self.alpha) * (all_y - all_hi) * (all_y > all_hi)
                ).mean()
            )

            summary = BacktestSummary(
                model_name=model_name,
                mae=mae,
                rmse=rmse,
                smape=smape,
                mase=mase,
                coverage=coverage,
                target_coverage=1 - self.alpha,
                mean_width=mean_width,
                winkler_score=winkler,
                n_forecasts=len(all_y),
                fold_results=folds,
            )
            rows.append(summary.as_dict())

        return pd.DataFrame(rows)

    def get_folds(self, model_name: str) -> List[FoldResult]:
        return self._results.get(model_name, [])

    def all_predictions(self) -> pd.DataFrame:
        """
        Return a long-form DataFrame of all fold predictions
        (useful for plotting).
        """
        records = []
        for model_name, folds in self._results.items():
            for fold in folds:
                for i, ts in enumerate(fold.actuals.index):
                    records.append(
                        {
                            "date": ts,
                            "origin": fold.origin,
                            "horizon_step": i + 1,
                            "model": model_name,
                            "actual": fold.actuals.iloc[i],
                            "forecast": fold.forecasts.iloc[i],
                            "lower": fold.lower.iloc[i],
                            "upper": fold.upper.iloc[i],
                        }
                    )
        return pd.DataFrame(records)
