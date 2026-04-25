#!/usr/bin/env python
import numpy as np
import pandas as pd

from pdslasso import PDSLasso


def _make_fe_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "d": [0, 1, 0, 1, 0, 1],
            "x0": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "x1": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "fe": pd.Categorical([0, 0, 1, 1, 2, 2]),
        }
    )


def test_partial_out_series_matches_group_mean() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"], fixed_effect_col="fe")
    fe_matrix = model._build_fixed_effects()
    y_resid = model._partial_out(df["y"], fe_matrix)
    group_mean = df.groupby("fe", observed=False)["y"].transform("mean")
    expected = df["y"] - group_mean
    assert np.allclose(y_resid.to_numpy(), expected.to_numpy())


def test_partial_out_dataframe_matches_group_mean() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"], fixed_effect_col="fe")
    fe_matrix = model._build_fixed_effects()
    values = df[["x0", "x1"]]
    x_resid = model._partial_out(values, fe_matrix)
    group_means = df.groupby("fe", observed=False)[["x0", "x1"]].transform("mean")
    expected = values - group_means
    assert np.allclose(x_resid.to_numpy(), expected.to_numpy())


def test_partial_out_no_fe_returns_input() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    y_resid = model._partial_out(df["y"], None)
    assert np.allclose(y_resid.to_numpy(), df["y"].to_numpy())


def test_fixed_effects_are_numeric() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"], fixed_effect_col="fe")
    fe_matrix = model._build_fixed_effects()
    assert fe_matrix is not None
    assert all(dtype.kind in ("i", "u", "f") for dtype in fe_matrix.dtypes)


def main() -> None:
    tests = [
        test_partial_out_series_matches_group_mean,
        test_partial_out_dataframe_matches_group_mean,
        test_partial_out_no_fe_returns_input,
        test_fixed_effects_are_numeric,
    ]
    for test in tests:
        test()
        print("OK:", test.__name__)


if __name__ == "__main__":
    main()
