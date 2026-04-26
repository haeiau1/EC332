# EC332 Time Series Project

This project contains the cleaned, reproducible version of the assignment.

## Structure

- `data/`: FRED CSV files used in the analysis.
- `scripts/`: Python scripts that reproduce the results.
- `outputs/part1_cpi_index/`: Part 1 tables, graphs, transformed data, and workbook.
- `outputs/part2_gdp_var/`: Part 2 VAR tables, graphs, forecasts, IRFs, and workbook.
- `reports/`: Markdown reports for submission/review.

## Run

From the project root:

```bash
.venv/bin/python scripts/part1_cpi_austria_index.py
.venv/bin/python scripts/part2_gdp_var_usa_austria.py
```

## Compile the LaTeX Report

From the project root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=reports reports/hw1_hayrettin_akguc_2020300168.tex
```

## Correct Data Files

- `AUTCPIALLMINMEI.csv`: Austria CPI index, monthly, not seasonally adjusted.
- `USAGDPRQPSMEI.csv`: USA quarterly real GDP growth rate.
- `AUTGDPRQPSMEI.csv`: Austria quarterly real GDP growth rate.

The earlier CPI growth-rate files were removed because Part 1 must start from a price index and calculate year-over-year inflation manually.
