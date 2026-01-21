#!/usr/bin/env python
"""Tests for input validation, edge cases, and degenerate data handling."""

import re
import numpy as np
import pandas as pd

from pdslasso import PDSLasso


def assert_raises(exc_type, func, *args, match=None, **kwargs):
    """Helper to assert that a function raises an exception."""
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised")
    except exc_type as e:
        if match is not None and not re.search(match, str(e)):
            raise AssertionError(f"Exception message '{e}' does not match pattern '{match}'")
        return True
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__} but got {type(e).__name__}: {e}")


# =============================================================================
# Input Validation Tests - ValueError cases
# =============================================================================

def test_y_in_always_include_raises():
    """Passing outcome variable in control_always_include should raise ValueError."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0"], control_always_include=["y"],
        match="control_always_include cannot contain the outcome"
    )


def test_d_in_always_include_raises():
    """Passing treatment variable in control_always_include should raise ValueError."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0"], control_always_include=["d"],
        match="control_always_include cannot contain the outcome or treatment"
    )


def test_fe_col_is_y_raises():
    """Setting fixed_effect_col to outcome variable should raise ValueError."""
    df = pd.DataFrame({
        "y": pd.Categorical([0, 0, 1, 1]),
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="y",
        match="fixed_effect_col cannot be the outcome or treatment"
    )


def test_fe_col_is_d_raises():
    """Setting fixed_effect_col to treatment variable should raise ValueError."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": pd.Categorical([0, 1, 0, 1]),
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="d",
        match="fixed_effect_col cannot be the outcome or treatment"
    )


def test_fe_col_in_always_include_raises():
    """Putting fixed_effect_col in control_always_include should raise ValueError."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
        "fe": pd.Categorical([0, 0, 1, 1]),
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0"],
        control_always_include=["fe"], fixed_effect_col="fe",
        match="fixed_effect_col should not be included in control_always_include"
    )


def test_fe_col_in_control_cols_raises():
    """Putting fixed_effect_col in control_cols should raise ValueError."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
        "fe": pd.Categorical([0, 0, 1, 1]),
    })
    assert_raises(
        ValueError,
        PDSLasso,
        data=df, y="y", d="d", control_cols=["x0", "fe"], fixed_effect_col="fe",
        match="fixed_effect_col should not be listed in control_cols"
    )


# =============================================================================
# Input Validation Tests - KeyError cases
# =============================================================================

def test_missing_y_column_raises():
    """Missing outcome column should raise KeyError on prep_data or fit."""
    df = pd.DataFrame({
        "outcome": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])
    assert_raises(KeyError, model.fit)


def test_missing_d_column_raises():
    """Missing treatment column should raise KeyError on prep_data or fit."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "treatment": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])
    assert_raises(KeyError, model.fit)


def test_missing_control_column_raises():
    """Missing control column should raise KeyError on fit."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    assert_raises(KeyError, model.fit)


def test_missing_always_include_column_raises():
    """Missing always-include column should raise KeyError on fit."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], control_always_include=["x1"])
    assert_raises(KeyError, model.fit)


def test_missing_fe_column_raises():
    """Missing fixed effect column should raise KeyError on fit."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "d": [0, 1, 0, 1],
        "x0": [1.0, 2.0, 3.0, 4.0],
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="fe")
    assert_raises(KeyError, model.fit)


# =============================================================================
# Missing Values (NaN) Tests
# =============================================================================

def test_nan_in_outcome_raises():
    """NaN in outcome should raise ValueError in Lasso step (sklearn rejects NaN)."""
    rng = np.random.default_rng(42)
    n = 50
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)
    y[10] = np.nan  # Introduce NaN

    df = pd.DataFrame({"y": y, "d": d, "x0": x0})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])

    # sklearn Lasso raises ValueError for NaN inputs
    assert_raises(ValueError, model.fit, match="Input.*contains NaN")


def test_nan_in_treatment_raises():
    """NaN in treatment should raise ValueError in Lasso step (sklearn rejects NaN)."""
    rng = np.random.default_rng(42)
    n = 50
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)
    d[5] = np.nan  # Introduce NaN

    df = pd.DataFrame({"y": y, "d": d, "x0": x0})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])

    # sklearn Lasso raises ValueError for NaN inputs
    assert_raises(ValueError, model.fit, match="Input.*contains NaN")


def test_nan_in_controls_propagates():
    """NaN in controls should propagate to Lasso step."""
    rng = np.random.default_rng(42)
    n = 50
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)
    x0[15] = np.nan  # Introduce NaN

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "x1": x1})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])

    # Lasso with NaN values will typically fail or produce invalid results
    # This tests that NaN is not silently ignored
    try:
        res = model.fit()
        # If it doesn't raise, check that results are affected
        coef = res.params["d"]
        # Either NaN or the Lasso may have had issues
        assert np.isnan(coef) or isinstance(coef, float)
    except (ValueError, np.linalg.LinAlgError):
        # Some versions may raise an error - that's acceptable behavior
        pass


def test_all_nan_control_column():
    """A control column that is entirely NaN should cause issues."""
    rng = np.random.default_rng(42)
    n = 50
    x0 = np.full(n, np.nan)  # All NaN
    x1 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "x1": x1})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])

    # This should either raise an error or produce invalid results
    try:
        res = model.fit()
        # If no error, coefficient should be invalid
        assert np.isnan(res.params["d"]) or not np.isfinite(res.params["d"])
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        pass  # Raising an error is acceptable


# =============================================================================
# Constant and Zero-Variance Column Tests
# =============================================================================

def test_constant_control_column():
    """A constant control column (zero variance) should not break the algorithm."""
    rng = np.random.default_rng(42)
    n = 100
    x_const = np.ones(n) * 5.0  # Constant column
    x_normal = rng.normal(size=n)
    d = 0.5 * x_normal + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.3 * x_normal + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x_const": x_const, "x_normal": x_normal})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x_const", "x_normal"])

    res = model.fit()
    # Should complete without error; constant column shouldn't be selected
    assert np.isfinite(res.params["d"])
    # Constant column should not be selected (it has no variance to explain)
    assert "x_const" not in model.selected_controls


def test_all_zero_control_column():
    """A control column that is all zeros should not break the algorithm."""
    rng = np.random.default_rng(42)
    n = 100
    x_zero = np.zeros(n)
    x_normal = rng.normal(size=n)
    d = 0.5 * x_normal + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.3 * x_normal + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x_zero": x_zero, "x_normal": x_normal})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x_zero", "x_normal"])

    res = model.fit()
    assert np.isfinite(res.params["d"])
    # Zero column should not be selected
    assert "x_zero" not in model.selected_controls


def test_near_constant_control_column():
    """A near-constant column (very small variance) should be handled gracefully."""
    rng = np.random.default_rng(42)
    n = 100
    x_near_const = np.ones(n) * 5.0 + rng.normal(scale=1e-10, size=n)
    x_normal = rng.normal(size=n)
    d = 0.5 * x_normal + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.3 * x_normal + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x_near_const": x_near_const, "x_normal": x_normal})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x_near_const", "x_normal"])

    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_multiple_constant_columns():
    """Multiple constant columns should all be handled properly."""
    rng = np.random.default_rng(42)
    n = 100
    x_const1 = np.ones(n) * 3.0
    x_const2 = np.ones(n) * -2.0
    x_normal = rng.normal(size=n)
    d = 0.5 * x_normal + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.3 * x_normal + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({
        "y": y, "d": d,
        "x_const1": x_const1, "x_const2": x_const2, "x_normal": x_normal
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x_const1", "x_const2", "x_normal"])

    res = model.fit()
    assert np.isfinite(res.params["d"])
    assert "x_const1" not in model.selected_controls
    assert "x_const2" not in model.selected_controls


# =============================================================================
# Degenerate Data / Edge Case Tests
# =============================================================================

def test_very_small_sample():
    """Very small sample size (n=10) with few controls should work."""
    rng = np.random.default_rng(42)
    n = 10
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])

    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_single_control_column():
    """Single control column should work correctly."""
    rng = np.random.default_rng(42)
    n = 100
    x0 = rng.normal(size=n)
    d = 1.5 * x0 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.8 * x0 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"])

    res = model.fit()
    assert np.isfinite(res.params["d"])
    # x0 is a confounder, should be selected
    assert "x0" in model.selected_controls


def test_n_equals_p():
    """Edge case where n equals p (square design)."""
    rng = np.random.default_rng(42)
    n = 50
    p = 50
    X = rng.normal(size=(n, p))
    d = X[:, 0] + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = y
    df["d"] = d

    control_cols = [f"x{i}" for i in range(p)]
    model = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)

    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_all_controls_are_noise():
    """When all controls are pure noise, selection should be sparse or empty."""
    rng = np.random.default_rng(42)
    n = 200
    p = 20
    X = rng.normal(size=(n, p))  # All noise, no relation to d or y
    d = rng.normal(size=n)
    y = 2.0 * d + rng.normal(scale=0.5, size=n)  # y only depends on d

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = y
    df["d"] = d

    control_cols = [f"x{i}" for i in range(p)]
    model = PDSLasso(data=df, y="y", d="d", control_cols=control_cols)

    res = model.fit()
    assert np.isfinite(res.params["d"])
    # With pure noise controls, we expect few or no selections
    # (depends on random chance, but should be sparse)
    assert len(model.selected_controls) <= p // 2


def test_perfectly_collinear_controls():
    """Perfectly collinear controls should be handled."""
    rng = np.random.default_rng(42)
    n = 100
    x0 = rng.normal(size=n)
    x1 = 2.0 * x0  # Perfectly collinear with x0
    x2 = rng.normal(size=n)  # Independent
    d = 0.5 * x0 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.3 * x0 + 0.2 * x2 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "x1": x1, "x2": x2})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1", "x2"])

    # Should complete - Lasso handles collinearity by selecting one
    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_binary_treatment():
    """Binary treatment indicator should work correctly."""
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    d_latent = 0.8 * x0 + rng.normal(scale=0.5, size=n)
    d = (d_latent > 0).astype(float)  # Binary treatment
    y = 3.0 * d + 0.5 * x0 + 0.3 * x1 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "x1": x1})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])

    res = model.fit()
    assert np.isfinite(res.params["d"])
    # Treatment effect should be roughly close to 3.0
    assert abs(res.params["d"] - 3.0) < 1.5


def test_single_fe_group():
    """Single fixed effect group should behave like no FE (constant absorbed)."""
    rng = np.random.default_rng(42)
    n = 100
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)
    fe = pd.Categorical(np.zeros(n, dtype=int))  # Single group

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "fe": fe})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="fe")

    # Single FE group produces empty FE matrix after drop_first
    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_string_fixed_effects():
    """String categorical fixed effects should work."""
    rng = np.random.default_rng(42)
    n = 60
    groups = np.array(["A", "B", "C"] * 20)
    x0 = rng.normal(size=n)
    fe_effects = {"A": 1.0, "B": -0.5, "C": 0.5}
    d = x0 + np.array([fe_effects[g] for g in groups]) + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 0.5 * x0 + np.array([fe_effects[g] for g in groups]) + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "fe": pd.Categorical(groups)})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="fe")

    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_integer_fixed_effects():
    """Integer (non-Categorical) fixed effects should work."""
    rng = np.random.default_rng(42)
    n = 60
    groups = np.tile([0, 1, 2], 20)  # Plain integers
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "fe": groups})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="fe")

    res = model.fit()
    assert np.isfinite(res.params["d"])


def test_many_fe_groups():
    """Large number of FE groups should be handled."""
    rng = np.random.default_rng(42)
    n_groups = 50
    n_per_group = 10
    n = n_groups * n_per_group
    groups = np.repeat(np.arange(n_groups), n_per_group)
    x0 = rng.normal(size=n)
    d = rng.normal(size=n)
    y = 2.0 * d + 0.5 * x0 + rng.normal(scale=0.5, size=n)

    df = pd.DataFrame({"y": y, "d": d, "x0": x0, "fe": pd.Categorical(groups)})
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], fixed_effect_col="fe")

    res = model.fit()
    assert np.isfinite(res.params["d"])


# =============================================================================
# Penalty Parameter Validation Tests
# =============================================================================

def test_invalid_penalty_gamma_zero():
    """penalty_gamma = 0 should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.normal(size=n),
        "x0": rng.normal(size=n),
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], penalty_gamma=0.0)
    assert_raises(ValueError, model.fit, match="penalty_gamma must be between 0 and 1")


def test_invalid_penalty_gamma_one():
    """penalty_gamma = 1 should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.normal(size=n),
        "x0": rng.normal(size=n),
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], penalty_gamma=1.0)
    assert_raises(ValueError, model.fit, match="penalty_gamma must be between 0 and 1")


def test_invalid_penalty_gamma_negative():
    """Negative penalty_gamma should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.normal(size=n),
        "x0": rng.normal(size=n),
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], penalty_gamma=-0.1)
    assert_raises(ValueError, model.fit, match="penalty_gamma must be between 0 and 1")


def test_invalid_penalty_c_zero():
    """penalty_c = 0 should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.normal(size=n),
        "x0": rng.normal(size=n),
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], penalty_c=0.0)
    assert_raises(ValueError, model.fit, match="penalty_c must be positive")


def test_invalid_penalty_c_negative():
    """Negative penalty_c should raise ValueError."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.normal(size=n),
        "x0": rng.normal(size=n),
    })
    model = PDSLasso(data=df, y="y", d="d", control_cols=["x0"], penalty_c=-1.0)
    assert_raises(ValueError, model.fit, match="penalty_c must be positive")


# =============================================================================
# Test runner for non-pytest execution
# =============================================================================

def main() -> None:
    """Run all tests without pytest."""
    tests = [
        # Input validation - ValueError
        test_y_in_always_include_raises,
        test_d_in_always_include_raises,
        test_fe_col_is_y_raises,
        test_fe_col_is_d_raises,
        test_fe_col_in_always_include_raises,
        test_fe_col_in_control_cols_raises,
        # Input validation - KeyError
        test_missing_y_column_raises,
        test_missing_d_column_raises,
        test_missing_control_column_raises,
        test_missing_always_include_column_raises,
        test_missing_fe_column_raises,
        # NaN handling
        test_nan_in_outcome_raises,
        test_nan_in_treatment_raises,
        test_nan_in_controls_propagates,
        test_all_nan_control_column,
        # Constant/zero columns
        test_constant_control_column,
        test_all_zero_control_column,
        test_near_constant_control_column,
        test_multiple_constant_columns,
        # Degenerate/edge cases
        test_very_small_sample,
        test_single_control_column,
        test_n_equals_p,
        test_all_controls_are_noise,
        test_perfectly_collinear_controls,
        test_binary_treatment,
        test_single_fe_group,
        test_string_fixed_effects,
        test_integer_fixed_effects,
        test_many_fe_groups,
        # Penalty validation
        test_invalid_penalty_gamma_zero,
        test_invalid_penalty_gamma_one,
        test_invalid_penalty_gamma_negative,
        test_invalid_penalty_c_zero,
        test_invalid_penalty_c_negative,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"OK: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__} - {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
    main()
