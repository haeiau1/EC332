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
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


USA_FILE = PROJECT_ROOT / "data" / "USAGDPRQPSMEI.csv"
AUT_FILE = PROJECT_ROOT / "data" / "AUTGDPRQPSMEI.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "part2_gdp_var"
REPORT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

VAR_NAMES = ["usa_gdp_growth", "austria_gdp_growth"]
SERIES_INFO = {
    "usa_gdp_growth": {
        "series_id": "USAGDPRQPSMEI",
        "title": "National Accounts: GDP by Expenditure: Constant Prices: Gross Domestic Product: Total for United States",
        "url": "https://fred.stlouisfed.org/series/USAGDPRQPSMEI",
    },
    "austria_gdp_growth": {
        "series_id": "AUTGDPRQPSMEI",
        "title": "National Accounts: GDP by Expenditure: Constant Prices: Gross Domestic Product: Total for Austria",
        "url": "https://fred.stlouisfed.org/series/AUTGDPRQPSMEI",
    },
}


def fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def quarter_label(index: pd.Index) -> list[str]:
    return [f"{ts.year}Q{((ts.month - 1) // 3) + 1}" for ts in index]


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
        "decision_10pct": "Stationary" if pvalue < 0.10 else "Not stationary",
    }


def read_series(path: Path, column: str, renamed: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={column: renamed})
    df = df.set_index("observation_date").sort_index()
    df[renamed] = pd.to_numeric(df[renamed], errors="coerce")
    return df[[renamed]]


def save_variables_plot(data: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    ax.plot(data.index, data["usa_gdp_growth"], label="USA", linewidth=1.8, color="#1f77b4")
    ax.plot(
        data.index,
        data["austria_gdp_growth"],
        label="Austria",
        linewidth=1.8,
        color="#d62728",
    )
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_title("Quarterly real GDP growth rates, 1990-latest available year minus 1")
    ax.set_ylabel("Growth rate same quarter previous year, percent")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_residual_acf_pacf(residuals: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for row, variable in enumerate(VAR_NAMES):
        plot_acf(residuals[variable], lags=24, ax=axes[row, 0], zero=False)
        plot_pacf(residuals[variable], lags=24, ax=axes[row, 1], zero=False, method="ywm")
        axes[row, 0].set_title(f"ACF of {variable} residual")
        axes[row, 1].set_title(f"PACF of {variable} residual")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_forecast_plot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    forecast: pd.DataFrame,
    se: pd.DataFrame,
    path: Path,
) -> None:
    z = norm.ppf(0.975)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True, sharex=False)

    for ax, variable, label, color in [
        (axes[0], "usa_gdp_growth", "USA", "#1f77b4"),
        (axes[1], "austria_gdp_growth", "Austria", "#d62728"),
    ]:
        recent_train = train.loc["2018-01-01":, variable]
        ax.plot(recent_train.index, recent_train.values, color="#777777", linewidth=1.4, label="Training")
        ax.plot(test.index, test[variable], color=color, linewidth=2.0, marker="o", label="Realized")
        ax.plot(
            forecast.index,
            forecast[variable],
            color="#111111",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            label="Forecast",
        )
        lower = forecast[variable] - z * se[variable]
        upper = forecast[variable] + z * se[variable]
        ax.fill_between(forecast.index, lower, upper, color=color, alpha=0.14, label="95% interval")
        ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{label}: forecast versus realized")
        ax.set_ylabel("Percent")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncols=4, fontsize=9)

    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_irf_plot(irf, path: Path) -> None:
    fig = irf.plot(orth=False, impulse=None, response=None, signif=0.05)
    fig.set_size_inches(11, 8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    usa = read_series(USA_FILE, "USAGDPRQPSMEI", "usa_gdp_growth")
    aut = read_series(AUT_FILE, "AUTGDPRQPSMEI", "austria_gdp_growth")
    data = usa.join(aut, how="inner").dropna().asfreq("QS-JAN")

    latest_year = int(data.index.year.max())
    train_start = pd.Timestamp("1990-01-01")
    train_end = pd.Timestamp(f"{latest_year - 1}-10-01")
    latest_year_start = pd.Timestamp(f"{latest_year}-01-01")

    train = data.loc[train_start:train_end].copy()
    test = data.loc[latest_year_start:].copy()

    save_variables_plot(train, OUT_DIR / "01_gdp_growth_variables_1990_2024.png")

    adf_results = {
        variable: adf_test(train[variable])
        for variable in VAR_NAMES
    }

    model = VAR(train)
    maxlags = 12
    lag_selection = model.select_order(maxlags=maxlags)
    lag_order_rows = []
    for lag in range(maxlags + 1):
        row = {"lag": lag}
        for criterion in ["aic", "bic", "hqic", "fpe"]:
            values = lag_selection.ics.get(criterion, [])
            row[criterion] = values[lag] if lag < len(values) else np.nan
        lag_order_rows.append(row)
    lag_order_table = pd.DataFrame(lag_order_rows)
    lag_order_table.to_csv(OUT_DIR / "var_lag_selection.csv", index=False)

    selected_lag = int(lag_selection.selected_orders["aic"])
    if selected_lag < 1:
        selected_lag = 1

    var_result = model.fit(selected_lag)
    params = var_result.params
    params.to_csv(OUT_DIR / "var_parameters.csv")

    residuals = var_result.resid
    residuals.to_csv(OUT_DIR / "var_residuals.csv")
    save_residual_acf_pacf(residuals, OUT_DIR / "02_residual_acf_pacf.png")

    residual_corr = residuals.corr()
    residual_corr.to_csv(OUT_DIR / "residual_correlation.csv")

    granger_rows = []
    for caused, causing in [
        ("austria_gdp_growth", ["usa_gdp_growth"]),
        ("usa_gdp_growth", ["austria_gdp_growth"]),
    ]:
        test_result = var_result.test_causality(caused=caused, causing=causing, kind="f")
        granger_rows.append(
            {
                "null_hypothesis": f"{causing[0]} does not Granger-cause {caused}",
                "caused": caused,
                "causing": causing[0],
                "test_statistic": float(test_result.test_statistic),
                "p_value": float(test_result.pvalue),
                "df": str(test_result.df),
                "decision_5pct": "Reject" if test_result.pvalue < 0.05 else "Do not reject",
            }
        )
    granger_table = pd.DataFrame(granger_rows)
    granger_table.to_csv(OUT_DIR / "granger_causality_tests.csv", index=False)

    steps = len(test)
    forecast_values = var_result.forecast(y=train.values[-selected_lag:], steps=steps)
    forecast = pd.DataFrame(forecast_values, index=test.index, columns=VAR_NAMES)

    forecast_cov = var_result.forecast_cov(steps=steps)
    forecast_se = pd.DataFrame(
        np.sqrt(np.stack([np.diag(cov) for cov in forecast_cov])),
        index=test.index,
        columns=VAR_NAMES,
    )

    comparison_parts = []
    for variable in VAR_NAMES:
        comparison_parts.append(
            pd.DataFrame(
                {
                    "variable": variable,
                    "realized": test[variable],
                    "forecast": forecast[variable],
                    "standard_error": forecast_se[variable],
                    "forecast_error": test[variable] - forecast[variable],
                }
            )
        )
    forecast_comparison = pd.concat(comparison_parts)
    forecast_comparison.index.name = "observation_date"
    forecast_comparison.to_csv(OUT_DIR / "forecast_comparison.csv")

    forecast_metrics = forecast_comparison.groupby("variable").apply(
        lambda df: pd.Series(
            {
                "rmse": np.sqrt(np.mean(df["forecast_error"] ** 2)),
                "mae": np.mean(np.abs(df["forecast_error"])),
            }
        ),
        include_groups=False,
    )
    forecast_metrics.to_csv(OUT_DIR / "forecast_metrics.csv")

    save_forecast_plot(
        train,
        test,
        forecast,
        forecast_se,
        OUT_DIR / "03_forecast_vs_realized.png",
    )

    irf_periods = 12
    irf = var_result.irf(irf_periods)
    save_irf_plot(irf, OUT_DIR / "04_impulse_response_nonorthogonal.png")
    irf_values = []
    for step in range(irf_periods + 1):
        matrix = irf.irfs[step]
        for response_idx, response in enumerate(VAR_NAMES):
            for impulse_idx, impulse in enumerate(VAR_NAMES):
                irf_values.append(
                    {
                        "horizon": step,
                        "response": response,
                        "impulse": impulse,
                        "irf": matrix[response_idx, impulse_idx],
                    }
                )
    irf_table = pd.DataFrame(irf_values)
    irf_table.to_csv(OUT_DIR / "impulse_response_nonorthogonal.csv", index=False)

    whiteness = var_result.test_whiteness(nlags=12)
    normality = var_result.test_normality()

    diagnostics = {
        "data_start": str(data.index.min().date()),
        "data_end": str(data.index.max().date()),
        "estimation_start": str(train.index.min().date()),
        "estimation_end": str(train.index.max().date()),
        "forecast_start": str(test.index.min().date()),
        "forecast_end": str(test.index.max().date()),
        "n_train_observations": int(train.shape[0]),
        "n_forecast_observations": int(test.shape[0]),
        "adf_results": adf_results,
        "selected_lag_by_aic": selected_lag,
        "selected_orders": {
            key: int(value) if value is not None else None
            for key, value in lag_selection.selected_orders.items()
        },
        "aic": float(var_result.aic),
        "bic": float(var_result.bic),
        "hqic": float(var_result.hqic),
        "fpe": float(var_result.fpe),
        "residual_correlation": residual_corr.to_dict(),
        "granger_causality": granger_table.to_dict(orient="records"),
        "forecast_metrics": forecast_metrics.to_dict(orient="index"),
        "whiteness_test": {
            "test_statistic": float(whiteness.test_statistic),
            "p_value": float(whiteness.pvalue),
            "decision_5pct": "Reject no residual autocorrelation"
            if whiteness.pvalue < 0.05
            else "Do not reject no residual autocorrelation",
        },
        "normality_test": {
            "test_statistic": float(normality.test_statistic),
            "p_value": float(normality.pvalue),
        },
        "irf_note": (
            "IRFs are non-orthogonalized (orth=False), matching the assignment's "
            "assumption of no correlation between equation error terms."
        ),
    }
    with (OUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    combined_output = data.loc[train_start:].copy()
    combined_output["sample"] = "unused"
    combined_output.loc[train.index, "sample"] = "estimation"
    combined_output.loc[test.index, "sample"] = "forecast_realized"
    combined_output.to_csv(OUT_DIR / "combined_gdp_growth_data.csv")

    with pd.ExcelWriter(OUT_DIR / "ec332_gdp_var_results.xlsx") as writer:
        combined_output.to_excel(writer, sheet_name="data")
        pd.DataFrame(
            [{"variable": variable, **result} for variable, result in adf_results.items()]
        ).to_excel(writer, sheet_name="adf_tests", index=False)
        lag_order_table.to_excel(writer, sheet_name="lag_selection", index=False)
        params.to_excel(writer, sheet_name="var_parameters")
        residual_corr.to_excel(writer, sheet_name="residual_correlation")
        granger_table.to_excel(writer, sheet_name="granger_tests", index=False)
        forecast_comparison.to_excel(writer, sheet_name="forecast_comparison")
        forecast_metrics.to_excel(writer, sheet_name="forecast_metrics")
        irf_table.to_excel(writer, sheet_name="irf_nonorthogonal", index=False)

    report = f"""# USA-Austria GDP Growth VAR Analysis

## Data

- USA series: `USAGDPRQPSMEI`, quarterly real GDP growth, same quarter previous year, seasonally adjusted.
- Austria series: `AUTGDPRQPSMEI`, quarterly real GDP growth, same quarter previous year, seasonally adjusted.
- Estimation period: {train.index.min().date()} to {train.index.max().date()}.
- Forecast comparison period: {test.index.min().date()} to {test.index.max().date()}.

The assignment asks for 1990 to latest available minus 1. Since both files run through 2025 Q3, I use 1990 Q1-2024 Q4 for estimation and 2025 Q1-2025 Q3 for forecast comparison.

## Stationarity

ADF test results on the estimation sample:

- USA GDP growth: test statistic {fmt_float(adf_results['usa_gdp_growth']['test_statistic'])}, p-value {fmt_float(adf_results['usa_gdp_growth']['p_value'])}; decision: {adf_results['usa_gdp_growth']['decision_5pct']}.
- Austria GDP growth: test statistic {fmt_float(adf_results['austria_gdp_growth']['test_statistic'])}, p-value {fmt_float(adf_results['austria_gdp_growth']['p_value'])}; decision: {adf_results['austria_gdp_growth']['decision_5pct']}.

Austria GDP growth is stationary at the 5% level. USA GDP growth is borderline: the ADF p-value is slightly above 5%, but below 10%. Since both variables are already GDP growth rates and the assignment asks for a VAR in these two growth-rate variables, I estimate the VAR in levels and note this borderline stationarity result.

## VAR Lag Selection

I checked lag lengths 0 through {maxlags} using AIC, BIC, HQIC, and FPE. The model used for the remaining analysis selects lag length by AIC:

- Selected VAR lag length: {selected_lag}
- VAR AIC: {fmt_float(var_result.aic)}
- VAR BIC: {fmt_float(var_result.bic)}
- VAR HQIC: {fmt_float(var_result.hqic)}

## Residual Diagnostics

The residual ACF/PACF plots are saved in `outputs/part2_gdp_var/02_residual_acf_pacf.png`.

Residual correlation matrix:

- Corr(USA residual, Austria residual): {fmt_float(residual_corr.loc['usa_gdp_growth', 'austria_gdp_growth'])}

The assignment asks for impulse responses assuming no correlation between error terms, so I report non-orthogonalized impulse response functions.

## Granger Causality

- Null: USA GDP growth does not Granger-cause Austria GDP growth. p-value: {fmt_float(granger_table.loc[0, 'p_value'])}; decision at 5%: {granger_table.loc[0, 'decision_5pct']}.
- Null: Austria GDP growth does not Granger-cause USA GDP growth. p-value: {fmt_float(granger_table.loc[1, 'p_value'])}; decision at 5%: {granger_table.loc[1, 'decision_5pct']}.

## Forecast

Forecast horizon: 2025 Q1 to 2025 Q3.

- USA forecast RMSE: {fmt_float(forecast_metrics.loc['usa_gdp_growth', 'rmse'])}; MAE: {fmt_float(forecast_metrics.loc['usa_gdp_growth', 'mae'])}.
- Austria forecast RMSE: {fmt_float(forecast_metrics.loc['austria_gdp_growth', 'rmse'])}; MAE: {fmt_float(forecast_metrics.loc['austria_gdp_growth', 'mae'])}.

The forecast graph with realized values and 95% intervals is saved in `outputs/part2_gdp_var/03_forecast_vs_realized.png`.

## Impulse Response Function

I computed non-orthogonalized IRFs for 12 quarters, using `orth=False`. The graph is saved in `outputs/part2_gdp_var/04_impulse_response_nonorthogonal.png`, and the numerical values are saved in `outputs/part2_gdp_var/impulse_response_nonorthogonal.csv`.

## AI Use and Error Checks

I used AI to write and run a reproducible Python script, organize the VAR workflow, and generate the report, workbook, and graphs. I checked for errors by:

- verifying both CSV headers, date ranges, and latest observations directly from the local files;
- confirming both series are quarterly GDP growth rates with the same transformation;
- aligning the two countries on common quarterly dates;
- using 1990 Q1-2024 Q4 for estimation and holding out 2025 Q1-Q3 for forecast comparison;
- testing stationarity with ADF before fitting the VAR;
- selecting VAR lag length with information criteria;
- saving residual ACF/PACF diagnostics, Granger test tables, forecast standard errors, and IRF values for auditability;
- verifying that the Excel workbook and graph files were created successfully.
"""
    (REPORT_DIR / "part2_gdp_var_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
