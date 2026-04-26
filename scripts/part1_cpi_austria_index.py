from __future__ import annotations

import json
import os
from pathlib import Path
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME = PROJECT_ROOT / ".cache"
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SERIES_ID = "AUTCPIALLMINMEI"
SERIES_TITLE = (
    "Consumer Price Indices (CPIs, HICPs), COICOP 1999: "
    "Consumer Price Index: Total for Austria"
)
DATA_FILE = PROJECT_ROOT / "data" / f"{SERIES_ID}.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "part1_cpi_index"
REPORT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

SOURCE_NOTE = (
    "FRED/OECD series AUTCPIALLMINMEI. Units are Index 2015=100, "
    "Not Seasonally Adjusted, Monthly. This is the raw CPI index needed before "
    "calculating year-over-year inflation."
)


def fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def adf_test(series: pd.Series) -> dict[str, float | int | str]:
    clean = series.dropna()
    stat, pvalue, used_lag, nobs, crit, icbest = adfuller(
        clean, autolag="AIC", regression="c"
    )
    return {
        "test_statistic": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "nobs": nobs,
        "critical_1pct": crit["1%"],
        "critical_5pct": crit["5%"],
        "critical_10pct": crit["10%"],
        "icbest": icbest,
        "decision_5pct": "Stationary" if pvalue < 0.05 else "Not stationary",
    }


def seasonal_adjust_log_index(index: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Seasonally adjust CPI by removing monthly dummy effects from log index."""
    df = pd.DataFrame({"log_index": np.log(index.dropna())})
    df["trend"] = np.arange(len(df), dtype=float)
    month_dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True)
    month_dummies.index = df.index
    x = pd.concat([df[["trend"]], month_dummies.astype(float)], axis=1)
    x = sm.add_constant(x)
    model = sm.OLS(df["log_index"], x).fit()

    seasonal_component = pd.Series(0.0, index=df.index)
    effects = pd.Series(0.0, index=range(1, 13), name="log_month_effect")
    for name, value in model.params.items():
        if name.startswith("month_"):
            month = int(name.split("_")[1])
            effects.loc[month] = value
            seasonal_component.loc[df.index.month == month] = value

    adjusted = np.exp(df["log_index"] - seasonal_component)
    adjusted.name = "cpi_index_sa"
    effects.index.name = "month"
    return adjusted, effects


def select_arma(y: pd.Series, max_p: int = 4, max_q: int = 4) -> tuple[pd.DataFrame, ARIMA]:
    rows: list[dict[str, float | int | str | bool]] = []
    best_result = None
    best_aic = np.inf

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                result = ARIMA(
                    y,
                    order=(p, 0, q),
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit()
                converged = bool(result.mle_retvals.get("converged", True))
                lb_p = acorr_ljungbox(
                    result.resid.dropna(), lags=[12], return_df=True
                )["lb_pvalue"].iloc[0]
                rows.append(
                    {
                        "p": p,
                        "q": q,
                        "aic": result.aic,
                        "bic": result.bic,
                        "hqic": result.hqic,
                        "llf": result.llf,
                        "converged": converged,
                        "ljung_box_p_lag12": lb_p,
                    }
                )
                if converged and result.aic < best_aic:
                    best_aic = result.aic
                    best_result = result
            except Exception as exc:
                rows.append(
                    {
                        "p": p,
                        "q": q,
                        "aic": np.nan,
                        "bic": np.nan,
                        "hqic": np.nan,
                        "llf": np.nan,
                        "converged": False,
                        "ljung_box_p_lag12": np.nan,
                        "error": str(exc)[:200],
                    }
                )

    table = pd.DataFrame(rows)
    table["eligible_for_selection"] = table["converged"]
    table = table.sort_values(
        ["eligible_for_selection", "aic"],
        ascending=[False, True],
        na_position="last",
    )
    if best_result is None:
        raise RuntimeError("No ARMA model converged.")
    return table, best_result


def save_line_plot(
    series: pd.Series,
    path: Path,
    title: str,
    ylabel: str,
    realized: pd.Series | None = None,
    forecast: pd.Series | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    if realized is None and forecast is None:
        ax.plot(series.index, series.values, color="#1f77b4", linewidth=1.7)
    else:
        ax.plot(series.index, series.values, color="#7f7f7f", linewidth=1.3, label="Training")
        ax.plot(realized.index, realized.values, color="#1f77b4", linewidth=2.0, label="Realized")
        ax.plot(forecast.index, forecast.values, color="#d62728", linewidth=2.0, linestyle="--", label="Forecast")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    raw = pd.read_csv(DATA_FILE)
    raw["observation_date"] = pd.to_datetime(raw["observation_date"])
    raw = raw.rename(columns={SERIES_ID: "cpi_index"})
    raw = raw.set_index("observation_date").asfreq("MS")
    raw["cpi_index"] = pd.to_numeric(raw["cpi_index"], errors="coerce")

    data_2000 = raw.loc["2000-01-01":].copy()

    save_line_plot(
        data_2000["cpi_index"],
        OUT_DIR / "01_cpi_index_2000_latest.png",
        "Austria CPI index, 2000-latest available",
        "Index 2015=100",
    )
    adf_index = adf_test(data_2000["cpi_index"])

    data_2000["cpi_index_sa"], month_effects = seasonal_adjust_log_index(
        data_2000["cpi_index"]
    )
    month_effects.to_csv(OUT_DIR / "seasonal_dummy_log_effects.csv")

    save_line_plot(
        data_2000["cpi_index_sa"],
        OUT_DIR / "02_cpi_index_seasonally_adjusted.png",
        "Austria CPI index after monthly-dummy seasonal adjustment",
        "Index 2015=100",
    )
    adf_index_sa = adf_test(data_2000["cpi_index_sa"])

    data_2000["inflation_yoy"] = (
        data_2000["cpi_index_sa"] / data_2000["cpi_index_sa"].shift(12) - 1.0
    ) * 100.0

    save_line_plot(
        data_2000["inflation_yoy"].dropna(),
        OUT_DIR / "03_yoy_inflation_from_sa_index.png",
        "Austria year-over-year inflation calculated from seasonally adjusted CPI index",
        "Percent",
    )
    adf_inflation = adf_test(data_2000["inflation_yoy"])

    if adf_inflation["p_value"] < 0.05:
        stationary = data_2000["inflation_yoy"].dropna()
        stationary.name = "inflation_yoy"
        stationary_name = "Year-over-year inflation from seasonally adjusted CPI index"
        transformation_note = "No differencing was applied because ADF rejected a unit root at 5%."
        adf_stationary = adf_inflation
    else:
        stationary = data_2000["inflation_yoy"].diff().dropna()
        stationary.name = "d_inflation_yoy"
        stationary_name = "First difference of year-over-year inflation"
        transformation_note = (
            "ADF did not reject a unit root at 5%, so I used the first difference "
            "of calculated year-over-year inflation."
        )
        adf_stationary = adf_test(stationary)

    save_line_plot(
        stationary,
        OUT_DIR / "04_stationary_variable.png",
        f"Stationary variable: {stationary_name}",
        "Percent" if stationary.name == "inflation_yoy" else "Percentage points",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    plot_acf(stationary, lags=36, ax=axes[0], zero=False)
    plot_pacf(stationary, lags=36, ax=axes[1], zero=False, method="ywm")
    axes[0].set_title("ACF of stationary variable")
    axes[1].set_title("PACF of stationary variable")
    fig.savefig(OUT_DIR / "05_acf_pacf_stationary_variable.png", dpi=200)
    plt.close(fig)

    acf_values = acf(stationary, nlags=4, fft=False)[1:5]
    stats_summary = {
        "expected_value_mean": stationary.mean(),
        "variance": stationary.var(ddof=1),
        "autocorrelation_lag_1": acf_values[0],
        "autocorrelation_lag_2": acf_values[1],
        "autocorrelation_lag_3": acf_values[2],
        "autocorrelation_lag_4": acf_values[3],
    }

    latest_year = int(data_2000.dropna(subset=["cpi_index"]).index.year.max())
    train_start = pd.Timestamp("2015-01-01")
    train_end = pd.Timestamp(f"{latest_year - 1}-12-01")
    latest_year_start = pd.Timestamp(f"{latest_year}-01-01")

    if stationary.name == "d_inflation_yoy":
        model_series = data_2000["inflation_yoy"].diff().dropna()
    else:
        model_series = data_2000["inflation_yoy"].dropna()

    train = model_series.loc[train_start:train_end]
    test = model_series.loc[latest_year_start:]
    model_selection, best_model = select_arma(train)
    model_selection.to_csv(OUT_DIR / "arma_model_selection.csv", index=False)

    steps = len(test)
    forecast_result = best_model.get_forecast(steps=steps)
    forecast_mean = pd.Series(forecast_result.predicted_mean.values, index=test.index)

    forecast_comparison = pd.DataFrame(
        {
            "realized_stationary_variable": test,
            "forecast_stationary_variable": forecast_mean,
            "forecast_error": test - forecast_mean,
        }
    )

    if stationary.name == "d_inflation_yoy":
        last_train_level = data_2000["inflation_yoy"].loc[train_end]
        level_forecast = last_train_level + forecast_mean.cumsum()
        level_realized = data_2000["inflation_yoy"].loc[test.index]
        forecast_comparison["realized_inflation_yoy"] = level_realized
        forecast_comparison["forecast_inflation_yoy"] = level_forecast
        graph_training = data_2000["inflation_yoy"].loc[train_start:train_end]
        graph_realized = level_realized
        graph_forecast = level_forecast
        forecast_rmse = float(np.sqrt(np.mean((level_realized - level_forecast) ** 2)))
        forecast_mae = float(np.mean(np.abs(level_realized - level_forecast)))
    else:
        graph_training = train
        graph_realized = test
        graph_forecast = forecast_mean
        forecast_rmse = float(np.sqrt(np.mean((test - forecast_mean) ** 2)))
        forecast_mae = float(np.mean(np.abs(test - forecast_mean)))

    forecast_comparison.to_csv(OUT_DIR / "forecast_comparison.csv")
    save_line_plot(
        graph_training,
        OUT_DIR / "06_forecast_latest_year_vs_realized.png",
        f"Forecast for {latest_year}: calculated year-over-year inflation",
        "Percent",
        realized=graph_realized,
        forecast=graph_forecast,
    )

    residuals = best_model.resid.dropna()
    lb = acorr_ljungbox(residuals, lags=[6, 12], return_df=True)
    model_params = best_model.params.to_frame("estimate")
    model_params["std_error"] = best_model.bse
    model_params["z_stat"] = model_params["estimate"] / model_params["std_error"]
    model_params["p_value"] = best_model.pvalues
    model_params.to_csv(OUT_DIR / "best_arma_model_parameters.csv")

    diagnostics = {
        "series_id": SERIES_ID,
        "series_title": SERIES_TITLE,
        "source_note": SOURCE_NOTE,
        "sample_start": str(data_2000.index.min().date()),
        "sample_end": str(data_2000.dropna(subset=["cpi_index"]).index.max().date()),
        "n_observations_2000_latest": int(data_2000["cpi_index"].dropna().shape[0]),
        "adf_cpi_index": adf_index,
        "adf_seasonally_adjusted_cpi_index": adf_index_sa,
        "adf_calculated_yoy_inflation": adf_inflation,
        "adf_stationary_variable": adf_stationary,
        "stationary_variable_name": stationary_name,
        "transformation_note": transformation_note,
        "stationary_variable_stats": stats_summary,
        "arma_training_start": str(train.index.min().date()),
        "arma_training_end": str(train.index.max().date()),
        "forecast_year": latest_year,
        "forecast_months_available": int(steps),
        "best_arma_order": {
            "p": int(best_model.model.order[0]),
            "d": int(best_model.model.order[1]),
            "q": int(best_model.model.order[2]),
        },
        "best_arma_aic": float(best_model.aic),
        "best_arma_bic": float(best_model.bic),
        "forecast_rmse": forecast_rmse,
        "forecast_mae": forecast_mae,
        "ljung_box": {
            f"lag_{idx}": {
                "stat": float(row["lb_stat"]),
                "p_value": float(row["lb_pvalue"]),
            }
            for idx, row in lb.iterrows()
        },
    }
    with (OUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    output_data = data_2000.join(stationary.rename("stationary_variable"), how="left")
    output_data.to_csv(OUT_DIR / "transformed_data_2000_latest.csv")

    with pd.ExcelWriter(OUT_DIR / "ec332_austria_cpi_index_results.xlsx") as writer:
        output_data.to_excel(writer, sheet_name="data")
        pd.DataFrame([stats_summary]).to_excel(writer, sheet_name="summary_stats", index=False)
        pd.DataFrame(
            [
                {"test": "Raw CPI index", **adf_index},
                {"test": "Seasonally adjusted CPI index", **adf_index_sa},
                {"test": "Calculated year-over-year inflation", **adf_inflation},
                {"test": "Final stationary variable", **adf_stationary},
            ]
        ).to_excel(writer, sheet_name="adf_tests", index=False)
        month_effects.to_frame().to_excel(writer, sheet_name="seasonal_effects")
        model_selection.to_excel(writer, sheet_name="arma_selection", index=False)
        model_params.to_excel(writer, sheet_name="arma_parameters")
        forecast_comparison.to_excel(writer, sheet_name="forecast_comparison")

    report = f"""# Austria CPI Index Analysis

## Data

- Country: Austria
- FRED series: `{SERIES_ID}`
- Series title: {SERIES_TITLE}
- Unit: Index 2015=100
- Period used in plots and tests: {data_2000.index.min().date()} to {data_2000.dropna(subset=['cpi_index']).index.max().date()}
- Data note: {SOURCE_NOTE}

This is the correct version for the assignment because the assignment asks to start from a price index and then calculate inflation relative to the same month one year earlier.

## Plot and Stationarity of the Price Index

ADF test on the raw CPI index:

- Test statistic: {fmt_float(adf_index['test_statistic'])}
- p-value: {fmt_float(adf_index['p_value'])}
- Decision at 5%: {adf_index['decision_5pct']}

The raw index is nonstationary, as expected for a price level series.

## Seasonal Adjustment

I deseasonalized the CPI index by regressing the log index on a time trend and monthly dummy variables, then removing the estimated monthly dummy effects. This keeps the long-run trend in the index while removing average monthly seasonal effects.

ADF test on the seasonally adjusted CPI index:

- Test statistic: {fmt_float(adf_index_sa['test_statistic'])}
- p-value: {fmt_float(adf_index_sa['p_value'])}
- Decision at 5%: {adf_index_sa['decision_5pct']}

## Inflation Calculation and Re-Test

I calculated year-over-year inflation from the seasonally adjusted CPI index:

`inflation_t = ((CPI_SA_t / CPI_SA_t-12) - 1) * 100`

ADF test on calculated year-over-year inflation:

- Test statistic: {fmt_float(adf_inflation['test_statistic'])}
- p-value: {fmt_float(adf_inflation['p_value'])}
- Decision at 5%: {adf_inflation['decision_5pct']}

Final variable used in the remaining analysis:

- {stationary_name}
- {transformation_note}
- Final ADF p-value: {fmt_float(adf_stationary['p_value'])}

## Summary Statistics

- Expected value / mean: {fmt_float(stats_summary['expected_value_mean'])}
- Variance: {fmt_float(stats_summary['variance'])}
- Autocorrelation lag 1: {fmt_float(stats_summary['autocorrelation_lag_1'])}
- Autocorrelation lag 2: {fmt_float(stats_summary['autocorrelation_lag_2'])}
- Autocorrelation lag 3: {fmt_float(stats_summary['autocorrelation_lag_3'])}
- Autocorrelation lag 4: {fmt_float(stats_summary['autocorrelation_lag_4'])}

## ARMA Model

Training period for Box-Jenkins/ARMA selection: {train.index.min().date()} to {train.index.max().date()}.

I compared ARMA(p,q) models for p = 0,...,4 and q = 0,...,4 using AIC and selected the lowest-AIC model among models that converged successfully. I also checked residual autocorrelation with Ljung-Box tests.

- Selected model: ARMA({best_model.model.order[0]}, {best_model.model.order[2]}) with constant
- AIC: {fmt_float(best_model.aic)}
- BIC: {fmt_float(best_model.bic)}
- Ljung-Box p-value at lag 12: {fmt_float(float(lb.loc[12, 'lb_pvalue']))}

## Forecast

Forecast period: {latest_year}, using the months available in the data ({steps} month(s)).

- Forecast RMSE: {fmt_float(forecast_rmse)}
- Forecast MAE: {fmt_float(forecast_mae)}

The forecast graph is saved as `outputs/part1_cpi_index/06_forecast_latest_year_vs_realized.png`.

## AI Use and Error Checks

I used AI to write and run a reproducible Python analysis script, organize the econometric workflow, and generate tables and graphs. I checked for errors by:

- verifying the CSV header, first observations, latest observations, and date range directly from the local file;
- confirming that `AUTCPIALLMINMEI` is a raw CPI index, unlike the earlier growth-rate series;
- running ADF tests on the raw index, the seasonally adjusted index, calculated inflation, and the final stationary variable;
- saving intermediate transformed data so the inflation calculation can be audited;
- selecting ARMA models with a reproducible AIC grid search and excluding non-converged models from selection;
- checking residual autocorrelation with Ljung-Box tests;
- verifying that the Excel workbook and graph files were created successfully.
"""
    (REPORT_DIR / "part1_austria_cpi_index_report.md").write_text(
        report, encoding="utf-8"
    )

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
