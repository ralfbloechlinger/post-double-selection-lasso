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


def _make_location_invariance_data(
    n: int = 300,
    seed: int = 20260718,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    controls = rng.normal(size=(n, 8))
    d_latent = (
        -1.5
        + 1.4 * controls[:, 0]
        - 0.9 * controls[:, 1]
        + rng.normal(size=n)
    )
    d = (d_latent > 0).astype(int)
    y = (
        4.0
        + 1.5 * d
        + 1.2 * controls[:, 2]
        - 0.8 * controls[:, 3]
        + rng.normal(size=n)
    )
    control_cols = [f"x{i}" for i in range(controls.shape[1])]
    return pd.DataFrame(controls, columns=control_cols).assign(d=d, y=y)


def _selected_stage_controls(
    model: PDSLasso,
    stage_name: str,
    control_cols: list[str],
) -> set[str]:
    stage = getattr(model, stage_name)
    return {
        control
        for control, coefficient in zip(control_cols, stage.coef_)
        if coefficient != 0
    }


def _assert_location_invariance(lasso_penalty_cv: bool) -> None:
    df = _make_location_invariance_data()
    control_cols = [column for column in df if column.startswith("x")]
    shifted = df.copy()
    shifted[control_cols] = shifted[control_cols] + 100.0
    shifted["y"] = shifted["y"] + 100.0

    base_model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=control_cols,
        lasso_penalty_cv=lasso_penalty_cv,
    )
    base_result = base_model.fit()
    shifted_model = PDSLasso(
        data=shifted,
        y="y",
        d="d",
        control_cols=control_cols,
        lasso_penalty_cv=lasso_penalty_cv,
    )
    shifted_result = shifted_model.fit()

    for stage_name in ("first_stage_lasso", "second_stage_lasso"):
        assert _selected_stage_controls(
            base_model, stage_name, control_cols
        ) == _selected_stage_controls(shifted_model, stage_name, control_cols)
    assert set(base_model.selected_controls) == set(shifted_model.selected_controls)
    assert np.isclose(
        base_result.params["d"],
        shifted_result.params["d"],
        rtol=1e-8,
        atol=1e-8,
    )


def test_location_invariance_with_parametric_penalty() -> None:
    _assert_location_invariance(lasso_penalty_cv=False)


def test_location_invariance_with_cross_validated_penalty() -> None:
    _assert_location_invariance(lasso_penalty_cv=True)


def test_constant_control_is_removed_by_intercept_residualization() -> None:
    df = _make_location_invariance_data()
    df["x_const"] = 5.0
    control_cols = [column for column in df if column.startswith("x")]
    model = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)

    result = model.fit()

    assert np.isfinite(result.params["d"])
    assert "x_const" not in model.selected_controls


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
    assert model.selected_controls == ["x0", "x1"]


def test_selected_controls_preserve_input_order(monkeypatch) -> None:
    rng = np.random.default_rng(20260718)
    n = 100
    df = pd.DataFrame({
        "x3": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "z": rng.normal(size=n),
    })
    df["d"] = 0.6 * df["x3"] + rng.normal(size=n)
    df["y"] = 1.5 * df["d"] + 0.4 * df["x2"] + rng.normal(size=n)
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=["x3", "x1", "x2"],
        control_always_include=["x1", "z"],
    )
    stage_selections = iter([
        ["x2", "x3"],
        ["x3"],
    ])

    def fake_run_lasso(*, X_ctrl, y_vec, feature_names):
        return None, next(stage_selections)

    monkeypatch.setattr(model, "_run_lasso", fake_run_lasso)
    result = model.fit()

    assert model.selected_controls == ["x3", "x1", "x2", "z"]
    assert list(result.params.index) == ["const", "d", "x3", "x1", "x2", "z"]


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


def test_summary_hides_fixed_effect_rows() -> None:
    df = _make_fe_zero_control_data()
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=["x_group", "x_noise"],
        fixed_effect_col="fe",
    )
    res = model.fit()
    summary_text = res.summary().as_text()

    assert "fe_1" not in summary_text
    assert "fe_2" not in summary_text
    assert "fe_3" not in summary_text
    assert "fe_4" not in summary_text
    assert "fe_5" not in summary_text
    assert "OLS with PDS-selected variables and full regressor set." in summary_text
    assert "Standard errors and t statistics valid for the following variables only: d." in summary_text
    assert "Includes FE for fe (5 absorbed dummies)." in summary_text


def test_summary_handles_multiple_fixed_effect_columns() -> None:
    df = _make_fe_zero_control_data()
    df = df.copy()
    df["fe_alt"] = pd.Categorical((df["x_noise"] > df["x_noise"].median()).astype(int))
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=["x_group", "x_noise"],
        fixed_effect_col=["fe", "fe_alt"],
    )
    res = model.fit()
    summary_text = res.summary().as_text()

    assert "fe_1" not in summary_text
    assert "fe_alt_1" not in summary_text
    assert "OLS with PDS-selected variables and full regressor set." in summary_text
    assert "Standard errors and t statistics valid for the following variables only: d." in summary_text
    assert "Includes FE for fe (5 absorbed dummies)." in summary_text
    assert "Includes FE for fe_alt (1 absorbed dummies)." in summary_text


def main() -> None:
    tests = [
        test_no_controls_simple_ols,
        test_empty_lasso_cols_keeps_always_include,
        test_scaling_invariance_selection_and_coef,
        test_location_invariance_with_parametric_penalty,
        test_location_invariance_with_cross_validated_penalty,
        test_constant_control_is_removed_by_intercept_residualization,
        test_p_gt_n_stress,
        test_zeroed_control_after_partial_out,
        test_summary_hides_fixed_effect_rows,
        test_summary_handles_multiple_fixed_effect_columns,
    ]
    for test in tests:
        test()
        print("OK:", test.__name__)


if __name__ == "__main__":
    main()
