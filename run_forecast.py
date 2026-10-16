"""
ForeDemand – Main Computation Pipeline
---------------------------------------
Runs the full forecasting pipeline and exports results to JSON
for the interactive dashboard.

Usage:
    python run_forecast.py
"""

import json
import sys
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.upi_data import load_upi_data
from models.forecasters import SeasonalNaive, ETSModel, HybridForecaster
from models.conformal import ConformalForecaster
from models.backtest import Backtester


def nan_to_none(obj):
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def series_to_records(s: pd.Series, value_name: str = "value") -> list:
    return [
        {"date": str(idx.date()), value_name: round(float(v), 4)}
        for idx, v in s.items()
    ]


def df_to_records(df: pd.DataFrame) -> list:
    records = []
    for idx, row in df.iterrows():
        rec = {"date": str(idx.date())}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                val = None if (np.isnan(val) or np.isinf(val)) else round(float(val), 4)
            elif isinstance(val, (int, np.integer)):
                val = int(val)
            rec[col] = val
        records.append(rec)
    return records


def main():
    print("=" * 60)
    print("  ForeDemand – Calibrated UPI Demand Forecasting")
    print("=" * 60)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading UPI transaction data...")
    df = load_upi_data()
    series = df["transactions_mn"]
    print(f"      {len(series)} monthly observations | "
          f"{series.index[0].strftime('%b %Y')} to {series.index[-1].strftime('%b %Y')}")

    # ── 2. Train/Test Split ───────────────────────────────────────────────────
    test_horizon = 12   # hold out last 12 months for evaluation
    train_series = series.iloc[:-test_horizon]
    test_series  = series.iloc[-test_horizon:]
    print(f"\n[2/5] Split: train={len(train_series)} obs | test={len(test_series)} obs")

    models_proto = [SeasonalNaive(), ETSModel(), HybridForecaster()]

    # ── 3. Final Forecast (train → next 12 months) ────────────────────────────
    print("\n[3/5] Computing final forecasts with conformal intervals...")
    final_forecasts = {}
    conformal_widths = {}

    for model in models_proto:
        print(f"      Fitting {model.name}...")
        cf = ConformalForecaster(model, alpha=0.10, calib_frac=0.25)
        cf.fit(train_series, max_horizon=test_horizon)
        pred_df = cf.predict_with_intervals(test_horizon)
        final_forecasts[model.name] = pred_df
        conformal_widths[model.name] = cf.quantile_widths.to_dict()

    # ── 4. Backtesting ────────────────────────────────────────────────────────
    print("\n[4/5] Running rolling-origin backtest...")
    backtester = Backtester(
        series=series,
        min_train=36,
        horizon=6,
        step=3,
        alpha=0.10,
        calib_frac=0.25,
    )

    fresh_models = [SeasonalNaive(), ETSModel(), HybridForecaster()]
    backtester.run(fresh_models)
    summary_df = backtester.summarise()
    all_preds_df = backtester.all_predictions()

    print("\n  Backtest Summary:")
    # Print without special unicode chars
    for _, row in summary_df.iterrows():
        print(f"    {row['Model']}: MAE={row['MAE']}  MASE={row['MASE']}  Cov={row['Coverage']}")

    # ── 5. Coverage check (final test set) ───────────────────────────────────
    print("\n[5/5] Coverage check on final test set...")
    coverage_results = {}
    for model_name, pred_df in final_forecasts.items():
        aligned = pred_df.reindex(test_series.index)
        lo = aligned["lower"].values
        hi = aligned["upper"].values
        y  = test_series.values
        inside = (y >= lo) & (y <= hi)
        cov = float(inside.mean())
        coverage_results[model_name] = {
            "empirical_coverage": round(cov, 4),
            "target_coverage": 0.90,
            "n_covered": int(inside.sum()),
            "n_total": len(y),
            "calibrated": abs(cov - 0.90) < 0.15,
        }
        status = "[OK] CALIBRATED" if coverage_results[model_name]["calibrated"] else "[!!] OFF"
        print(f"      {model_name}: {cov*100:.1f}% coverage  {status}")

    # ── 6. Serialise to JSON ──────────────────────────────────────────────────
    output = {
        "meta": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "series": "India Monthly UPI Transactions (Millions)",
            "source": "NPCI – npci.org.in",
            "train_end": str(train_series.index[-1].date()),
            "test_start": str(test_series.index[0].date()),
            "test_end": str(test_series.index[-1].date()),
            "alpha": 0.10,
            "coverage_target": "90%",
        },
        "historical": series_to_records(series, "transactions_mn"),
        "train_series": series_to_records(train_series, "transactions_mn"),
        "test_series": series_to_records(test_series, "transactions_mn"),
        "forecasts": {},
        "backtest_summary": summary_df.to_dict(orient="records"),
        "coverage_check": coverage_results,
        "conformal_widths": conformal_widths,
    }

    for model_name, pred_df in final_forecasts.items():
        output["forecasts"][model_name] = df_to_records(
            pred_df[["forecast", "lower", "upper"]]
        )

    # Add all backtest predictions (sampled to avoid huge JSON)
    if not all_preds_df.empty:
        # Keep last-origin fold only for each model (most recent backtest)
        last_origin = all_preds_df.groupby("model")["origin"].max().to_dict()
        filtered = all_preds_df[
            all_preds_df.apply(lambda r: r["origin"] == last_origin[r["model"]], axis=1)
        ]
        output["latest_backtest_fold"] = nan_to_none(
            filtered.assign(
                date=filtered["date"].astype(str),
                origin=filtered["origin"].astype(str),
            ).to_dict(orient="records")
        )

    out_path = os.path.join(os.path.dirname(__file__), "dashboard", "data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(nan_to_none(output), f, indent=2, default=str)

    print(f"\n[OK] Results saved to: {out_path}")
    print("\nDone! Open dashboard/index.html in a browser to explore results.")


if __name__ == "__main__":
    main()
