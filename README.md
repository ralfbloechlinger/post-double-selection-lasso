# pds-lasso

A simple Python implementation of the post-double selection LASSO estimator for treatment effects with high-dimensional controls, as proposed by Belloni, Chernozhukov, and Hansen (2014).

## Features

- Post-double selection LASSO for partially linear models with a minimal class-based interface
- Supports binary and continuous treatment variables
- **Feasible Lasso with optimal penalty loadings** (default) as described in Belloni, Chernozhukov, and Hansen (2014)
- Uses `Lasso` or `LassoCV` from `scikit-learn` for the two selection steps
- Treats the intercept as unpenalized by partialling out a constant before both selection steps
- Penalty level based on the parametric choice in BCH (2014) by default, with optional cross-validation
- Uses `statsmodels.api.OLS` for the final unpenalized regression with HC1 or one-way cluster-robust standard errors
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
    cov_type="HC1",                      # Final-regression covariance type
    cluster_cov=None,                    # Column for one-way clustered inference
)
```

### Treatment variables

`d` may identify a binary or continuous treatment column. The column must use
a real numeric or Boolean dtype, contain only finite non-missing values, and
vary across observations. Common binary representations such as Boolean,
nullable Boolean, integer, nullable integer, and floating-point 0/1 values are
supported.

Treatment data are validated when `fit()` is called and converted on an
internal Series; the input DataFrame is not modified. The treatment must retain
variation after partialling out fixed effects and always-included controls, and
it must be identified in the final regression design. The outcome and treatment
must be distinct, and the treatment must not also be listed in `control_cols`,
`control_always_include`, or `fixed_effect_col`.

When `cluster_cov` is provided, it overrides `cov_type` for the final OLS
regression. The cluster column must exist, contain no missing identifiers, and
contain at least two distinct clusters. Conventional cluster-robust inference
can be unreliable with only a few clusters.

Clustering currently applies only to final-regression inference. The two Lasso
selection stages continue to use observation-level heteroskedastic penalty
loadings; they are not cluster-aware.

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
