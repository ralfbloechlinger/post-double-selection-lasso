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


def test_partial_out_without_controls_centers_series() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    y_resid = model._partial_out(df["y"], None)
    expected = df["y"] - df["y"].mean()
    assert isinstance(y_resid, pd.Series)
    assert y_resid.index.equals(df.index)
    assert y_resid.name == "y"
    assert np.allclose(y_resid.to_numpy(), expected.to_numpy())


def test_partial_out_without_controls_centers_matrix_inputs() -> None:
    df = _make_fe_data()
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    values = df[["x0", "x1"]]

    frame_resid = model._partial_out(values, None)
    array_resid = model._partial_out(values.to_numpy(), None)
    vector_resid = model._partial_out(values["x0"].to_numpy(), None)
    expected = values - values.mean()

    assert isinstance(frame_resid, pd.DataFrame)
    assert frame_resid.index.equals(values.index)
    assert frame_resid.columns.equals(values.columns)
    assert np.allclose(frame_resid.to_numpy(), expected.to_numpy())
    assert isinstance(array_resid, np.ndarray)
    assert array_resid.shape == values.shape
    assert np.allclose(array_resid, expected.to_numpy())
    assert isinstance(vector_resid, np.ndarray)
    assert vector_resid.shape == (len(values),)
    assert np.allclose(vector_resid, expected["x0"].to_numpy())


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
        test_partial_out_without_controls_centers_series,
        test_partial_out_without_controls_centers_matrix_inputs,
        test_fixed_effects_are_numeric,
    ]
    for test in tests:
        test()
        print("OK:", test.__name__)


if __name__ == "__main__":
    main()
