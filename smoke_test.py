#!/usr/bin/env python
from pdslasso import PDSLasso
from sim_data import simulate_pds_data


def main():
    df, _ = simulate_pds_data(n=1500, p=30, true_effect=2.0, random_seed=0)
    model = PDSLasso(
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


if __name__ == "__main__":
    main()
