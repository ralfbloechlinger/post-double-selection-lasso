## Model and sparse approximation

Partially linear (treatment effect) model:
$$
y_i = d_i \alpha_0 + g(z_i) + \zeta_i,\quad \mathbb{E}[\zeta_i\mid z_i,d_i]=0,
$$
$$
d_i = m(z_i) + v_i,\quad \mathbb{E}[v_i\mid z_i]=0.
$$

Let $x_i=P(z_i)\in\mathbb{R}^p$ be a (possibly very large) dictionary of controls/transformations, and approximate
$$
g(z_i)=x_i'\beta_{g0}+r_{gi},\qquad m(z_i)=x_i'\beta_{m0}+r_{mi},
$$
with **approximate sparsity**: $\|\beta_{g0}\|_0\le s$, $\|\beta_{m0}\|_0\le s$ for $s\ll n$ and approximation errors small relative to estimation error, e.g.
$$
\left(\mathbb{E}[\bar r_{gi}^2]\right)^{1/2}\lesssim \sqrt{s/n},\qquad
\left(\mathbb{E}[\bar r_{mi}^2]\right)^{1/2}\lesssim \sqrt{s/n}.
$$

## The PDS / post-double-selection algorithm

**Goal:** estimate $\alpha_0$ with valid (uniform) inference when $p$ can exceed $n$.

1. **Selection for the treatment equation.** Run a (feasible) Lasso of $d_i$ on $x_i$ and record selected indices
   $$
   \widehat I_1=\operatorname{support}(\widehat\beta_1).
   $$

2. **Selection for the outcome reduced form.** Run a (feasible) Lasso of $y_i$ on $x_i$ and record
   $$
   \widehat I_2=\operatorname{support}(\widehat\beta_2).
   $$

3. **Union (plus optional “amelioration” controls).** Form
   $$
   \widehat I=\widehat I_1\cup \widehat I_2\cup \widehat I_3,
   $$
   where $\widehat I_3$ is any small, user-chosen set of always-include controls.

4. **Post-selection OLS (the PDS estimator).** Regress $y_i$ on $d_i$ and $x_{i,\widehat I}$ by least squares:
   $$
   (\check\alpha,\check\beta)=\arg\min_{\alpha,\beta}\ \mathbb{E}_n[(y_i-d_i\alpha-x_i'\beta)^2]\quad
   \text{s.t. }\beta_j=0\ \forall j\notin\widehat I.
   $$

5. **Inference.** Use conventional OLS inference for $\check\alpha$ from Step 4 (heteroskedasticity-robust by default).

## “Feasible Lasso” details used for Steps 1–2

Each selection step solves a weighted Lasso of the form
$$
\min_{\beta\in\mathbb{R}^p}\ \mathbb{E}_n[(\tilde y_i-\tilde x_i'\beta)^2] + \lambda\,\|\widehat\Psi\beta\|_1,
\qquad \widehat\Psi=\mathrm{diag}(\widehat\ell_1,\ldots,\widehat\ell_p).
$$

A heteroskedasticity-robust (theory) choice uses
$$
\lambda = 2c\sqrt{n}\,\Phi^{-1}\!\left(1-\frac{\gamma}{2p}\right),
\qquad \widehat\ell_j\approx \ell_j:=\sqrt{\mathbb{E}_n[\tilde x_{ij}^2\varepsilon_i^2]},
$$
with user constants $c>1$ and $\gamma\in(0,1)$ (practical choices mentioned in the paper include $c=1.1$, $\gamma=0.05$).

**Iterated loadings (sketch):** start from a small initial model $I_0$ (e.g., intercept), compute residuals, update
$$
\widehat\ell_{j}^{(k+1)}=\sqrt{\mathbb{E}_n\!\left[x_{ij}^2\,(y_i-x_i'\widehat\beta^{(k)})^2\right]}\cdot\sqrt{\frac{n}{n-\widehat s_b}},
$$
and iterate until the loadings converge (tolerance $\nu$) or a max iteration cap $K$ is reached.

**Square-root Lasso option:** alternatively solve
$$
\min_{\beta}\ \sqrt{\mathbb{E}_n[(\tilde y_i-\tilde x_i'\beta)^2]} + \frac{\lambda}{n}\,\|\widehat\Psi\beta\|_1,
\qquad \lambda=c\sqrt{n}\,\Phi^{-1}\!\left(1-\frac{\gamma}{2p}\right),
$$
with loadings that can be simpler under homoskedasticity (and bounded/iterated under heteroskedasticity).
