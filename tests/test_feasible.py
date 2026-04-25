#!/usr/bin/env python
import numpy as np
import pandas as pd

from pdslasso import PDSLasso 
from pdslasso.sim_data import simulate_pds_data


def _make_signal_data(n: int = 500, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    d = 1.5 * x0 + 1.2 * x1 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 1.0 * x0 + 0.8 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def _make_fe_zero_control_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_groups = 6
    n_per = 30
    groups = np.repeat(np.arange(n_groups), n_per)
    n = groups.size
    x_group = groups.astype(float)
    x_noise = rng.normal(size=n)
    fe_d = rng.normal(scale=0.5, size=n_groups)
    fe_y = rng.normal(scale=0.5, size=n_groups)
    d = 0.8 * x_noise + fe_d[groups] + rng.normal(scale=0.5, size=n)
    y = 1.5 * d + 0.5 * x_noise + fe_y[groups] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x_group": x_group,
            "x_noise": x_noise,
            "fe": pd.Categorical(groups),
        }
    )


def test_no_controls_simple_ols() -> None:
    rng = np.random.default_rng(0)
    n = 200
    d = rng.normal(size=n)
    y = 2.5 * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d})
    model = PDSLasso(data=df, y="y", d="d", control_cols=None)
    res = model.fit()
    assert model.selected_controls == []
    assert "d" in res.params.index
    assert abs(res.params["d"] - 2.5) < 0.5


def test_empty_lasso_cols_keeps_always_include() -> None:
    rng = np.random.default_rng(1)
    n = 150
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    d = 1.2 * x0 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.7 * x1 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "x1": x1})
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=["x0", "x1"],
        control_always_include=["x0", "x1"],
    )
    model.fit()
    assert set(model.selected_controls) == {"x0", "x1"}


def test_scaling_invariance_selection_and_coef() -> None:
    df = _make_signal_data()
    control_cols = ["x0", "x1", "x2", "x3"]
    model_base = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)
    res_base = model_base.fit()

    df_scaled = df.copy()
    df_scaled["x1"] = df_scaled["x1"] * 0.1
    model_scaled = PDSLasso(data=df_scaled, y="y", d="d", control_cols=control_cols)
    res_scaled = model_scaled.fit()

    assert set(model_base.selected_controls) == set(model_scaled.selected_controls)
    assert abs(res_base.params["d"] - res_scaled.params["d"]) < 1e-6


def test_p_gt_n_stress() -> None:
    df, _ = simulate_pds_data(n=40, p=200, random_seed=321)
    control_cols = [c for c in df.columns if c.startswith("x")]
    model = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)
    res = model.fit()
    assert np.isfinite(res.params["d"])
    assert set(model.selected_controls).issubset(set(control_cols))


def test_zeroed_control_after_partial_out() -> None:
    df = _make_fe_zero_control_data()
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=["x_group", "x_noise"],
        fixed_effect_col="fe",
    )
    model.fit()
    assert "x_group" not in model.selected_controls


def main() -> None:
    tests = [
        test_no_controls_simple_ols,
        test_empty_lasso_cols_keeps_always_include,
        test_scaling_invariance_selection_and_coef,
        test_p_gt_n_stress,
        test_zeroed_control_after_partial_out,
    ]
    for test in tests:
        test()
        print("OK:", test.__name__)


if __name__ == "__main__":
    main()
