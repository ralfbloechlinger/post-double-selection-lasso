# pds-lasso

A simple Python implementation of the post-double selection LASSO estimator for treatment effects with high-dimensional controls, as proposed by Belloni, Chernozhukov, and Hansen (2014). 

## Features

- Post-double selection LASSO for partially linear models with a minimal class-based interface
- Uses `LassoCV` or `Lasso' from `scikit-learn` for the two selection steps  
- Penalisation hyperparameter ($\lambda$) based on the parametric choice in Belloni, Chernozhukov, and Hansen (2014) or CV
- Uses `statsmodels.api.OLS` for the final unpenalized regression  
- Allows for partialling out of fixed-effects (as categortical variables) and selected, always-included controls


## Open Issues: 
- Include "feasible lasso" (optimal penalty-term loadings) from paper 


## Requirements

* `pandas`
* `numpy`
* `scikit-learn`
* `statsmodels`

## Usage (minimal example)

```python
df = pd.read_csv("mydata.csv")

model = PDSLasso(
    data=df,
    y="outcome_var",
    d="treatment_var",
    control_cols=["x1", "x2", "x3", "x4"],
    lasso_penalty="paper"
)

results = model.fit()

print("Selected controls:", model.selected_controls)
print(results.summary())
```

## Remarks / Disclaimer

This repository is a basic personal implementation of post-double selection LASSO, shared in case it is useful to other researchers:

* It is *not* a production-ready econometrics package.
* No guarantees are made about correctness, numerical stability, or suitability for any particular empirical setting.
* Results may differ from reference implementations (e.g. `pdslasso` in Stata or `hdm` in R)


### Citation 
If you use this code in academic work, please cite the original methodological paper:

**Alexandre Belloni, Victor Chernozhukov, Christian Hansen**, Inference on Treatment Effects after Selection among High-Dimensional Controls, *The Review of Economic Studies*, Volume 81, Issue 2, April 2014, Pages 608–650, https://doi.org/10.1093/restud/rdt044

### No warranty / no liability

This software is provided **“as is”**, without any express or implied warranty. In no event shall the author be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or its use.

By using this code, you agree that you are responsible for verifying its suitability for your use case and for checking your results.

