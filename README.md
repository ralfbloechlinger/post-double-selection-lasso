# pds-lasso

A simple Python implementation of the post-double selection LASSO estimator for treatment effects with high-dimensional controls, as proposed by Belloni, Chernozhukov, and Hansen (2014).

## Features

- Post-double selection LASSO for partially linear models with a minimal class-based interface
- **Feasible Lasso with optimal penalty loadings** (default) as described in Belloni, Chernozhukov, and Hansen (2014)
- Uses `Lasso` or `LassoCV` from `scikit-learn` for the two selection steps
- Penalty level based on the parametric choice in BCH (2014) by default, with optional cross-validation
- Uses `statsmodels.api.OLS` for the final unpenalized regression with HC1 robust standard errors
- Supports partialling out of fixed effects (as categorical variables) and always-included controls

## Installation

```bash
pip install pdslasso
```

Or install from source:

```bash
git clone https://github.com/ralfblochlinger/post-double-selection-lasso.git
cd post-double-selection-lasso
pip install -e .
```

## Requirements

* `pandas`
* `numpy`
* `scikit-learn`
* `statsmodels`

## Usage (minimal example)

```python
from pdslasso import PDSLasso

df = pd.read_csv("mydata.csv")

model = PDSLasso(
    data=df,
    y="outcome_var",
    d="treatment_var",
    control_cols=["x1", "x2", "x3", "x4"],
)

results = model.fit()

print("Selected controls:", model.selected_controls)
print(results.summary())
```

### Options

```python
model = PDSLasso(
    data=df,
    y="outcome_var",
    d="treatment_var",
    control_cols=["x1", "x2", "x3", "x4"],
    control_always_include=["x1"],       # Controls always included (not penalized)
    fixed_effect_col="group",            # Fixed effects (partialled out)
    lasso_penalty_cv=False,              # Use parametric penalty (default) or CV
    penalty_c=1.1,                       # Constant for parametric penalty
    penalty_gamma=0.05,                  # Significance level for parametric penalty
)
```

## Remarks / Disclaimer

This repository is a basic personal implementation of post-double selection LASSO, shared in case it is useful to other researchers:

* It is *not* a production-ready econometrics package.
* No guarantees are made about correctness, numerical stability, or suitability for any particular empirical setting.
* Results may differ from reference implementations (e.g. `pdslasso` in Stata or `hdm` in R)


### Citation
If you use this code in academic work, please cite the original methodological paper:

**Alexandre Belloni, Victor Chernozhukov, Christian Hansen**, Inference on Treatment Effects after Selection among High-Dimensional Controls, *The Review of Economic Studies*, Volume 81, Issue 2, April 2014, Pages 608-650, https://doi.org/10.1093/restud/rdt044

### No warranty / no liability

This software is provided **"as is"**, without any express or implied warranty. In no event shall the author be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or its use.

By using this code, you agree that you are responsible for verifying its suitability for your use case and for checking your results.
