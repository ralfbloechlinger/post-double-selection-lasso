#!/usr/bin/env python
import pandas as pd

import pdslasso


def main():
    df, _ = pdslasso.sim_data.simulate_pds_data(n=1500, p=30, true_effect=2.0, random_seed=0)

    print("Paper penalty loading -------------")
    model = pdslasso.PDSLasso(
        data=df,
        y="y",
        d="d",
        lasso_penalty_cv=False,
        control_cols=[c for c in df.columns if c.startswith("x")],
    )
    results = model.fit()

    if model.selected_controls is None:
        raise RuntimeError("Expected selected controls, got None.")
    if not hasattr(results, "params"):
        raise RuntimeError("Expected statsmodels results with params.")

    print("OK: fitted model with", len(model.selected_controls), "selected controls.")
    print("Treatment coefficient:", results.params["d"])

    print("\nCross-validation penalty loading and always include -------------")
    model = pdslasso.PDSLasso(
        data=df,
        y="y",
        d="d",
        lasso_penalty_cv=True,
        control_always_include="x11",
        control_cols=[c for c in df.columns if c.startswith("x")],
    )
    results = model.fit()

    if model.selected_controls is None:
        raise RuntimeError("Expected selected controls, got None.")
    if not hasattr(results, "params"):
        raise RuntimeError("Expected statsmodels results with params.")

    print("OK: fitted model with", len(model.selected_controls), "selected controls.")
    print("Treatment coefficient:", results.params["d"])

    print("\nFixed effects data generation -------------")
    df_fe, _ = pdslasso.sim_data.simulate_pds_data(
        n=1500,
        p=30,
        true_effect=2.0,
        random_seed=1,
        include_fixed_effects=True,
        n_groups=12,
        fe_outcome_sd=4
    )
    model = pdslasso.PDSLasso(
        data=df_fe,
        y="y",
        d="d",
        lasso_penalty_cv=False,
        fixed_effect_col="fe",
        control_always_include="x11",
        control_cols=[c for c in df_fe.columns if c.startswith("x")],
    )
    results = model.fit()

    if model.selected_controls is None:
        raise RuntimeError("Expected selected controls, got None.")
    if not hasattr(results, "params"):
        raise RuntimeError("Expected statsmodels results with params.")

    print("OK: fitted model with", len(model.selected_controls), "selected controls.")
    print("Treatment coefficient:", results.params["d"])

    print("\nSummary output with fixed effects -------------")
    df_fe_small, _ = pdslasso.sim_data.simulate_pds_data(
        n=250,
        p=20,
        true_effect=2.0,
        random_seed=2,
        include_fixed_effects=True,
        n_groups=6,
        fe_outcome_sd=3,
    )
    model = pdslasso.PDSLasso(
        data=df_fe_small,
        y="y",
        d="d",
        lasso_penalty_cv=False,
        fixed_effect_col="fe",
        control_cols=[c for c in df_fe_small.columns if c.startswith("x")],
    )
    results = model.fit()
    print(results.summary())

    df_fe_multi = df_fe_small.copy()
    df_fe_multi["fe_alt"] = pd.Categorical(
        (df_fe_multi["x0"] > df_fe_multi["x0"].median()).astype(int)
    )
    model = pdslasso.PDSLasso(
        data=df_fe_multi,
        y="y",
        d="d",
        lasso_penalty_cv=False,
        fixed_effect_col=["fe", "fe_alt"],
        control_cols=[c for c in df_fe_multi.columns if c.startswith("x")],
    )
    results = model.fit()
    print(results.summary())

    print("\nFalsely excluding Fe from construction -------------")
    df_fe = df_fe.copy().drop(columns=["fe"])
    model = pdslasso.PDSLasso(
        data=df_fe,
        y="y",
        d="d",
        lasso_penalty_cv=False,
        control_cols=[c for c in df_fe.columns if c.startswith("x")],
    )
    results = model.fit()

    if model.selected_controls is None:
        raise RuntimeError("Expected selected controls, got None.")
    if not hasattr(results, "params"):
        raise RuntimeError("Expected statsmodels results with params.")

    print("OK: fitted model with", len(model.selected_controls), "selected controls.")
    print("Treatment coefficient:", results.params["d"])



if __name__ == "__main__":
    main()
