#!/usr/bin/env python
import csv
import os

import numpy as np
import pandas as pd

from pdslasso import PDSLasso


# Paths relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "pdslasso_sim.csv")
STATA_RESULTS_PATH = os.path.join(_PROJECT_ROOT, "stata", "pdslasso_results.csv")
COEF_TOL = 1e-3


def _parse_selected_controls(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if "," not in text:
        text = text.replace(" ", ",")
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


def _read_stata_results(path: str) -> tuple[float, list[str]]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError(f"Stata results file is empty: {path}")
    header = rows[0]
    data = rows[1]
    if "treatment_coef" not in header:
        raise ValueError(f"Stata results missing treatment_coef header: {path}")
    if "selected_controls" not in header:
        raise ValueError(f"Stata results missing selected_controls header: {path}")
    if not data:
        raise ValueError(f"Stata results data row is empty: {path}")
    d_coef = float(data[0])
    if len(data) == 2:
        selected = _parse_selected_controls(data[1])
    else:
        selected = [value.strip() for value in data[1:] if value.strip()]
    return d_coef, selected


def test_compare_stata_results() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    if not os.path.exists(STATA_RESULTS_PATH):
        raise FileNotFoundError(
            f"Missing Stata results file: {STATA_RESULTS_PATH}. "
            "Run stata/pdslasso_compare.do to generate it."
        )

    df = pd.read_csv(DATA_PATH)
    control_cols = [c for c in df.columns if c.startswith("x")]
    model = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)
    res = model.fit()
    print(res.summary())
    selected_py = set(model.selected_controls)
    d_coef_py = float(res.params["d"])

    d_coef_stata, selected_stata_list = _read_stata_results(STATA_RESULTS_PATH)
    selected_stata = set(selected_stata_list)

    if selected_py != selected_stata:
        raise AssertionError(
            f"Selected controls differ. Python={sorted(selected_py)} "
            f"Stata={sorted(selected_stata)}"
        )
    if not np.isclose(d_coef_py, d_coef_stata, atol=COEF_TOL, rtol=0.0):
        raise AssertionError(
            f"Treatment coefficient differs. Python={d_coef_py:.6f} "
            f"Stata={d_coef_stata:.6f} (tol={COEF_TOL})"
        )


def main() -> None:
    test_compare_stata_results()
    print("OK: test_compare_stata_results")


if __name__ == "__main__":
    main()
