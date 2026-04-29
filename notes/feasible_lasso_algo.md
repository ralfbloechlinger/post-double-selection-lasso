# Feasible Lasso with penalty loadings (Belloni–Chernozhukov–Hansen, 2014)

This document specifies an implementation-ready algorithm for the **feasible Lasso** (heteroskedasticity-robust Lasso with **iteratively estimated penalty loadings**) used in Belloni, Chernozhukov, and Hansen (2014), eq. (2.12) and Algorithm 1 in Appendix A.

The algorithm below is written to integrate cleanly with the current `pdslasso.py` structure, where `y`, `d`, and the candidate controls `X` are **residualized** (“partialled out”) with respect to: 
- always-include controls, and
- fixed effects (one-hot dummies),
via Frisch–Waugh–Lovell before the Lasso steps.

---

## 1. Notation and mapping to the codebase

Let:

- `y_vec` be the dependent variable in a Lasso step (either the residualized outcome `y_resid` or residualized treatment `d_resid`).
- `X_ctrl` be the candidate control matrix in that Lasso step (the residualized controls `X_lasso_resid`), with shape `(n, p)`.
- `feature_names[j]` name the `j`-th column of `X_ctrl`.

We want to solve the **weighted (loaded) Lasso** problem

$$
\min_{\beta\in\mathbb{R}^p} \; \frac{1}{n}\sum_{i=1}^n (y_i - x_i'\beta)^2
\; + \; \frac{\lambda}{n} \sum_{j=1}^p \ell_j |\beta_j|.
$$

where:
- `λ` is the penalty level (same for all coefficients),
- `ℓ_j` is the penalty loading for regressor `j` (estimated from the data).

---

## 2. Penalty level $\lambda$ (eq. 2.12)

For a given Lasso step with `n` observations and `p` candidate controls:

$$
\lambda = 2 c \sqrt{n}\; \Phi^{-1}\left(1 - \frac{\gamma}{2p}\right),
$$

where:
- `c` corresponds to `penalty_c` (typical default 1.1),
- `γ` corresponds to `penalty_gamma` (typical default 0.05),
- `Φ^{-1}` is the standard normal quantile function.

### Mapping to scikit-learn’s `Lasso(alpha=...)`

`sklearn.linear_model.Lasso` uses objective

$$
\frac{1}{2n}\|y - X\beta\|_2^2 + \alpha\|\beta\|_1.
$$

To match the paper’s objective, set:

$$
\alpha = \frac{\lambda}{2n}.
$$

(If you use the scaling trick in Section 3, the same `α` is used; the loadings enter through the design rescaling.)

---

## 3. Implementing loadings via column rescaling

A weighted penalty $\sum_j \ell_j |\beta_j|$ can be turned into an unweighted penalty by defining:

- rescaled design: $\tilde X_{ij} = X_{ij} / \ell_j$,
- transformed coefficients: $\theta_j = \ell_j \beta_j$,
- Note that these two aspects cancel out. Via insertion into the objective function we can see that the approach is correct.
- Due to rescaling of coefficients we do NOT want to standardise X before the algoirthm.

Then:

$$
y - X\beta = y - \tilde X\theta, \qquad
\sum_j \ell_j |\beta_j| = \sum_j |\theta_j|.
$$

So we can fit an **ordinary Lasso** on `X_scaled = X / loadings`, obtain `theta_hat`,
and map back:

$$
\beta\_hat = \theta\_hat / \ell.
$$

Implementation details:
- Let `loadings` be a length-`p` array.
- Compute `X_scaled = X_ctrl / loadings` **columnwise**.
- Fit `Lasso(alpha=lambda_/(2n), fit_intercept=False, max_iter=...)` to `(X_scaled, y_vec)`.
- Unscale coefficients to the original regressors: `beta_hat = theta_hat / loadings`.

> **Numerical stability:** enforce a floor, e.g. `loadings = np.maximum(loadings, eps)`.

---

## 4. Estimating penalty loadings (Algorithm 1, Appendix A)

### 4.1 Inputs and hyperparameters

Inputs per Lasso step:
- `y`: array shape `(n,)`
- `X`: array shape `(n, p)`
- `c, γ` (penalty hyperparameters)
- stopping parameters:
  - `K` (max iterations), e.g. 5–10
  - `ν` (tolerance), e.g. `1e-4` or `1e-3`
- `eps` loading floor, e.g. `1e-12`

### 4.2 Initialization (important)

Algorithm 1 initializes loadings from residuals of an initial “small” model `I0`.
In our integration, the clean choice is:

- If the code already **residualized `y` and `X`** with respect to fixed effects and always-include controls,
  take residuals from `I0 = {}` on the residualized data, i.e.:

  - `e0 = y` (since `y` is already partialled out)
  - `ℓ_{j,0} = sqrt( mean( X[:,j]^2 * e0^2 ) )`

- If you run feasible Lasso on *raw* (non-residualized) data, then a faithful choice is `I0 = {intercept}`,
  i.e. `e0 = y - mean(y)`.

### 4.3 Iteration

For `k = 0, 1, ..., K-1`:

1) **Compute penalty level**  
   `λ = 2*c*sqrt(n)*Phi^{-1}(1 - γ/(2p))`

2) **Weighted Lasso via rescaling**  
   - `X_scaled = X / ℓ_k` (columnwise)
   - `alpha = λ / (2n)`
   - Fit Lasso on `(X_scaled, y)` → `theta_hat`
   - Map back: `beta_hat = theta_hat / ℓ_k`
   - Selected set `T_k = { j : beta_hat[j] != 0 }`
   - Let `s_k = |T_k|`

3) **Post-Lasso (OLS refit on selected set)**  
   - If `T_k` is empty: set `e_k = y`
   - Else:
     - Solve OLS on the original scale using only selected columns:
       `beta_pl = argmin_b || y - X_T b ||_2^2`
     - Residuals: `e_k = y - X_T @ beta_pl`

   Notes:
   - Since `y` and `X` are residualized in the PDS workflow, **do not add an intercept** in Post-Lasso.
   - Use `np.linalg.lstsq` (or statsmodels OLS without constant) for numerical robustness.

4) **Update loadings with df correction**  
   Compute the raw loading update:

   $$
   \tilde\ell_{j,k+1} = \sqrt{ \frac{1}{n}\sum_{i=1}^n x_{ij}^2 e_{k,i}^2 }.
   $$

   Apply the degrees-of-freedom correction from Algorithm 1:

   $$
   \ell_{j,k+1} = \tilde\ell_{j,k+1}\;\sqrt{\frac{n}{\max(n - s_k,\; 1)}}.
   $$

   Then clip:
   - `ℓ_{k+1} = np.maximum(ℓ_{k+1}, eps)`.

5) **Check convergence**  
   - Compute `delta = max_j |ℓ_{j,k+1} - ℓ_{j,k}|`.
   - If `delta <= ν`, stop and return `ℓ_{k+1}`.

Output:
- final loadings `ℓ_hat`
- optionally: the final Lasso fit and selected set from the last iteration.

### 4.4 Pseudocode

```python
def feasible_loadings(y, X, c=1.1, gamma=0.05, K=6, nu=1e-4, eps=1e-12):
    n, p = X.shape

    # init residuals e0
    e = y  # if y already partialled out; else use y - y.mean()

    # init loadings
    l = np.sqrt(np.mean((X**2) * (e[:, None]**2), axis=0))
    l = np.maximum(l, eps)

    for k in range(K):
        # penalty level
        lam = 2 * c * np.sqrt(n) * norm.ppf(1 - gamma/(2*p))
        alpha = lam / (2*n)

        # weighted lasso via rescaling
        Xs = X / l
        theta = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000).fit(Xs, y).coef_
        beta = theta / l
        T = np.flatnonzero(beta != 0)
        s = len(T)

        # post-lasso residuals
        if s == 0:
            e = y
        else:
            XT = X[:, T]
            b_pl, *_ = np.linalg.lstsq(XT, y, rcond=None)
            e = y - XT @ b_pl

        # loading update + df correction
        l_new = np.sqrt(np.mean((X**2) * (e[:, None]**2), axis=0))
        l_new *= np.sqrt(n / max(n - s, 1))
        l_new = np.maximum(l_new, eps)

        if np.max(np.abs(l_new - l)) <= nu:
            return l_new

        l = l_new

    return l
```

---

## 5. Integration into `pdslasso.py` (post-double selection)

In `fit()` (current structure):

1) Prepare data: `y_vec`, `d_vec`, candidate controls `X_lasso`.
2) Build `partial_out_matrix` (always-include controls + FE dummies).
3) Residualize:
   - `X_lasso_resid = partial_out(X_lasso, partial_out_matrix)`
   - `y_resid = partial_out(y_vec, partial_out_matrix)`
   - `d_resid = partial_out(d_vec, partial_out_matrix)`

Then run **two feasible-Lasso selections**:

- Lasso step A (treatment model): feasible Lasso of `d_resid` on `X_lasso_resid`
  → selected set `T_d`

- Lasso step B (outcome model): feasible Lasso of `y_resid` on `X_lasso_resid`
  → selected set `T_y`

Union:
- `T = T_d ∪ T_y ∪ always_include_controls`

Final regression (unchanged from existing code):
- OLS of **raw** `y` on **raw** `d` and selected controls (plus FE dummies),
  with `cov_type="HC1"`.

---

## 6. Implementation notes and edge cases

- **No candidate controls (`p = 0`)**: skip feasible Lasso and return empty selection.
- **All-zero / near-zero columns** after partial-out can produce near-zero loadings:
  - clip `ℓ_j` to `eps` and optionally drop columns whose variance is numerically zero.
- **fit_intercept**:
  - In the PDS workflow (after partial-out), set `fit_intercept=False` for both Lasso and Post-Lasso.
- **Determinism**:
  - sklearn’s coordinate descent is deterministic given the same inputs (no random seed needed),
    but upstream preprocessing should preserve column ordering and alignment.

---

## 7. Recommended defaults

Good “paper-like” defaults:
- `penalty_c = 1.1`
- `penalty_gamma = 0.05`
- `K = 6`
- `nu = 1e-4` (or `1e-3` if you want fewer iterations)
- `eps = 1e-12`

