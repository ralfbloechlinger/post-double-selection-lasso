"""Post-double-selection Lasso estimator for a single treatment effect.

Implements the procedure described in Belloni, Chernozhukov, and Hansen (2014)
to select controls via two Lasso fits and then estimate the treatment effect
with OLS using heteroskedasticity-robust (HC1) standard errors.

Algorithm overview:
1) Feasible Lasso: d on X with penalty loadings to select controls predictive of treatment.
2) Feasible Lasso: y on X with penalty loadings to select controls predictive of outcome.
3) OLS: y on d and the union of selected controls.

Example:
    from pdslasso import PDSLasso
    est = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    res = est.fit()
"""

import math
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


class PDSData(NamedTuple):
    y: pd.Series
    d: pd.Series
    X: pd.DataFrame | None


class PDSLasso:
    """Post-double-selection Lasso estimator for a single treatment effect."""
    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        d: str,
        control_cols: list[str] | None = None,
        control_always_include: list[str] | str | None = None,
        fixed_effect_col: str | list[str] | None = None,
        lasso_penalty_cv: bool = False,
        penalty_c: float = 1.1,
        penalty_gamma: float = 0.05,
        penalty_sigma: float | None = None,
        feasible_lasso_max_iter: int = 6,
        feasible_lasso_tol: float = 1e-4,
        feasible_lasso_eps: float = 1e-12,
        cov_type: str | None = "HC1" ,
        cluster_cov: str | None = None,
    ):
        """Initialize the estimator with data columns and penalty settings.

        Args:
            data: Input DataFrame containing outcome, treatment, and controls.
            y: Column name for the outcome variable.
            d: Column name for the treatment variable.
            control_cols: Column names for candidate controls. If None, fit OLS
                with only the treatment variable.
            control_always_include: Column names for controls that should always be
                included in the final OLS step (I3 in the paper) and partialed
                out in the Lasso steps. A single column name can be passed as a
                string.
            fixed_effect_col: Optional column name for fixed effects. This column
                is one-hot-encoded and included as a non-penalized control in
                both Lasso steps (via partialing out) and the final OLS step.
            lasso_penalty_cv: If True, use cross-validation to select the Lasso
                penalty; otherwise use the parametric penalty from Belloni et al.
            penalty_c: Positive constant for the parametric penalty formula.
            penalty_gamma: Value in (0, 1) for the parametric penalty formula.
            penalty_sigma: Optional fixed sigma for the parametric penalty.
                Currently unused with feasible Lasso; retained for compatibility.
            feasible_lasso_max_iter: Maximum iterations for feasible Lasso loadings.
            feasible_lasso_tol: Tolerance for feasible Lasso loading convergence.
            feasible_lasso_eps: Floor to avoid zero penalty loadings.
            cov_type: Type of covariance matrix to use for inference.
            cluster_cov: Optional column name for clustering variable for standard errors, overrides cov_type if provided.
        Raises:
            KeyError: If any of y, d, or control_cols are missing from data.
        """
        self.data = data
        self.y = y 
        self.d = d
        self.control_cols = control_cols
        if control_always_include is None:
            self.control_always_include = []
        elif isinstance(control_always_include, str):
            self.control_always_include = [control_always_include]
        else:
            self.control_always_include = control_always_include
        if fixed_effect_col is None:
            self.fixed_effect_col = []
        elif isinstance(fixed_effect_col, str):
            self.fixed_effect_col = [fixed_effect_col]

        self.lasso_penalty = lasso_penalty_cv
        self.penalty_c = penalty_c
        self.penalty_gamma = penalty_gamma
        self.penalty_sigma = penalty_sigma
        self.feasible_lasso_max_iter = feasible_lasso_max_iter
        self.feasible_lasso_tol = feasible_lasso_tol
        self.feasible_lasso_eps = feasible_lasso_eps

        if any(col in self.control_always_include for col in [self.y, self.d]):
            raise ValueError("control_always_include cannot contain the outcome or treatment variable.")
        if any(col in [self.y, self.d] for col in self.fixed_effect_col):
            raise ValueError("fixed_effect_col cannot be the outcome or treatment variable.")
        if any(
            fe_col in self.control_always_include for fe_col in self.fixed_effect_col
        ):
            raise ValueError("fixed_effect_col should not be included in control_always_include.")
        if self.control_cols is not None and any(
            fe_col in self.control_cols for fe_col in self.fixed_effect_col):
            raise ValueError("fixed_effect_col should not be listed in control_cols; pass it separately.")

    def __repr__(self) -> str:
        repr_str = f"PDS-Lasso class with dependent variable {self.y}, variable of interest {self.d}"
        return repr_str
    
    def prep_data(self) -> PDSData:
        """Prepare and return the response, treatment, and optional controls."""
        y_vec = self.data[self.y]
        d_vec = self.data[self.d]
        X_ctrl = None if self.control_cols is None else self.data[self.control_cols]
        return PDSData(y=y_vec, d=d_vec, X=X_ctrl)

    def _build_fixed_effects(self) -> pd.DataFrame | None:
        """
        Convert fixed effect column(s) to one-hot-encoded DataFrame.
        """
        if not self.fixed_effect_col:
            return None

        all_fe_dummies = []
        for fe_col_name in self.fixed_effect_col:
            fe_raw = self.data[fe_col_name]
            if not pd.api.types.is_categorical_dtype(fe_raw):
                fe_raw = fe_raw.astype("category")
            dummies = pd.get_dummies(fe_raw, prefix=fe_col_name, drop_first=True).astype(float)
            if dummies.shape[1] > 0:
                all_fe_dummies.append(dummies)

        if not all_fe_dummies:
            return None
        return pd.concat(all_fe_dummies, axis=1)

    def _build_partial_out_matrix(
        self,
        control_always_include: list[str],
        fe_matrix: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """
        Build matrix of variables to partial out in Lasso steps: 
        - always-include controls
        - fixed effects
        """
        pieces = []
        if control_always_include:
            pieces.append(self.data[control_always_include])
        if fe_matrix is not None:
            pieces.append(fe_matrix)
        if not pieces:
            return None
        return pd.concat(pieces, axis=1)

    def _partial_out(
        self,
        values: pd.Series | pd.DataFrame | np.ndarray,
        fe_matrix: pd.DataFrame | None,
    ) -> pd.Series | pd.DataFrame | np.ndarray:
        """
        Partial out fixed effects and always-include controls from values using OLS. 
        (controls must be included in fe matrix)
        Logic follows from Frisch-Waugh-Lovell.
        Values may be a matrix or vector (DataFrame, Series, or ndarray).
        """
        if fe_matrix is None:
            return values
        fe_design = sm.add_constant(fe_matrix, has_constant="add")
        design = fe_design.to_numpy()
        if isinstance(values, pd.Series):
            y_mat = values.to_numpy().reshape(-1, 1)
            coef, _, _, _ = np.linalg.lstsq(design, y_mat, rcond=None)
            resid = y_mat - design @ coef
            return pd.Series(resid.ravel(), index=values.index, name=values.name)
        if isinstance(values, pd.DataFrame):
            y_mat = values.to_numpy()
            coef, _, _, _ = np.linalg.lstsq(design, y_mat, rcond=None)
            resid = y_mat - design @ coef
            return pd.DataFrame(resid, index=values.index, columns=values.columns)
        y_mat = np.asarray(values)
        if y_mat.ndim == 1:
            y_mat = y_mat.reshape(-1, 1)
        coef, _, _, _ = np.linalg.lstsq(design, y_mat, rcond=None)
        resid = y_mat - design @ coef
        if resid.shape[1] == 1:
            return resid.ravel()
        return resid
         

    def _estimate_sigma(self, y_vec: np.ndarray | pd.Series) -> float:
        """Estimate sigma for the parametric penalty if not provided."""
        if self.penalty_sigma is not None:
            return float(self.penalty_sigma)
        return float(np.std(y_vec, ddof=1))

    def _penalty_level(self, n_obs: int, n_ctrl: int) -> float:
        """Compute parametric penalty level from Belloni et al. (2014), eq. (2.12)."""
        if not 0 < self.penalty_gamma < 1:
            raise ValueError("penalty_gamma must be between 0 and 1.")
        if self.penalty_c <= 0:
            raise ValueError("penalty_c must be positive.")
        if n_ctrl <= 0:
            raise ValueError("n_ctrl must be positive.")
        quantile = NormalDist().inv_cdf(1 - self.penalty_gamma / (2 * n_ctrl))
        return 2 * self.penalty_c * math.sqrt(n_obs) * quantile

    def _post_lasso_residuals(
        self,
        X_ctrl: np.ndarray,
        y_vec: np.ndarray,
        selected_idx: np.ndarray,
    ) -> np.ndarray:
        """Compute post-Lasso residuals without intercept on selected controls."""
        if selected_idx.size == 0:
            return y_vec
        X_sel = X_ctrl[:, selected_idx]
        coef, _, _, _ = np.linalg.lstsq(X_sel, y_vec, rcond=None)
        return y_vec - X_sel @ coef

    def _run_lasso(
        self,
        X_ctrl: np.ndarray | pd.DataFrame,
        y_vec: np.ndarray | pd.Series,
        feature_names: list[str],
    ) -> tuple[Lasso | LassoCV, list[str]]:
        """Run feasible Lasso with penalty loadings and return fitted model and selected controls."""
        X_mat = np.asarray(X_ctrl)
        y_arr = np.asarray(y_vec).ravel()
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(-1, 1)
        n_obs, n_ctrl = X_mat.shape
        if n_ctrl == 0:
            return Lasso(alpha=0.0), []

        # initialize loadings using residualized y
        loadings = np.sqrt(np.mean((X_mat ** 2) * (y_arr[:, None] ** 2), axis=0))
        loadings = np.maximum(loadings, self.feasible_lasso_eps)

        # initialize now to handle 0 iteration case 
        last_fit: Lasso | LassoCV | None = None
        last_beta = np.zeros(n_ctrl) 
        # iterate until loadings converge or max iterations are reached
        for _ in range(self.feasible_lasso_max_iter):
            X_scaled = X_mat / loadings
            if self.lasso_penalty:
                lasso_fit = LassoCV(fit_intercept=False, max_iter=10000).fit(
                    X=X_scaled,
                    y=y_arr,
                )
                theta_hat = lasso_fit.coef_
            else:
                penalty_level = self._penalty_level(n_obs, n_ctrl)
                alpha = penalty_level / (2 * n_obs)
                lasso_fit = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000).fit(
                    X=X_scaled,
                    y=y_arr,
                )
                theta_hat = lasso_fit.coef_

            beta_hat = theta_hat / loadings
            selected_idx = np.flatnonzero(beta_hat)
            s_k = selected_idx.size
            resid = self._post_lasso_residuals(X_mat, y_arr, selected_idx)

            loadings_new = np.sqrt(np.mean((X_mat ** 2) * (resid[:, None] ** 2), axis=0))
            loadings_new *= math.sqrt(n_obs / max(n_obs - s_k, 1))
            loadings_new = np.maximum(loadings_new, self.feasible_lasso_eps)

            last_fit = lasso_fit
            last_beta = beta_hat
            if np.max(np.abs(loadings_new - loadings)) <= self.feasible_lasso_tol:
                loadings = loadings_new
                break
            loadings = loadings_new

        selected = [col for col, c in zip(feature_names, last_beta) if c != 0]
        if last_fit is None:
            last_fit = Lasso(alpha=0.0, fit_intercept=False)
        return last_fit, selected
    
    def fit(self) -> RegressionResultsWrapper:
        """Fit the post-double-selection model and return the final regression.

        Returns:
            Statsmodels OLS results with HC1 standard errors.

        Raises:
            ValueError: If penalty settings are invalid when using the parametric
                penalty (see _run_lasso).
        """

        data = self.prep_data()
        y_vec = data.y
        d_vec = data.d

        # build partialling out matrix to residualise y,d,X for Lasso steps
        # partialling out always-include controls and fixed effects
        control_always_include = list(self.control_always_include)
        fe_matrix = self._build_fixed_effects()
        partial_out_matrix = self._build_partial_out_matrix(control_always_include, fe_matrix)

        # no controls => simple OLS
        if self.control_cols is None:
            selected_conts = control_always_include
            selected_vars = [self.d] + selected_conts

        else:
            # controls for lasso (excluding always-include controls)
            lasso_cols = [col for col in self.control_cols if col not in control_always_include]
            X_lasso = data.X[lasso_cols]

            # Residualise each of X,y,d with respect to FE and always included controls
            X_lasso_resid = self._partial_out(X_lasso, partial_out_matrix)
            y_resid = self._partial_out(y_vec, partial_out_matrix)
            d_resid = self._partial_out(d_vec, partial_out_matrix)

            
            if lasso_cols:
                # Lasso 1: treatment indicator on all other controls 
                lasso_1, selected_cols_d_on_X = self._run_lasso(
                    X_ctrl=X_lasso_resid,
                    y_vec=d_resid,
                    feature_names=lasso_cols,
                )
                # Lasso 2: outcome on all other controls
                lasso_2, selected_cols_y_on_X = self._run_lasso(
                    X_ctrl=X_lasso_resid,
                    y_vec=y_resid,
                    feature_names=lasso_cols,
                )
                
            else:
                lasso_1 = None
                lasso_2 = None
                selected_cols_d_on_X = []
                selected_cols_y_on_X = []

            self.first_stage_lasso = lasso_1
            self.second_stage_lasso = lasso_2 

            # selected controls as union of both sets of controls
            selected_conts = list(set(selected_cols_d_on_X + selected_cols_y_on_X + control_always_include))
            selected_vars = [self.d] + selected_conts

        # final matrix of X: variable of interest plus selected contrs
        X_final_df = self.data[selected_vars]
        if fe_matrix is not None:
            X_final_df = pd.concat([X_final_df, fe_matrix], axis=1)
        X_final_vec = sm.add_constant(X_final_df, has_constant="add")

        # fit object
        fin_reg = sm.OLS(y_vec, X_final_vec)
        if self.cluster_cov is not None:
            fin_reg_fit = fin_reg.fit(cov_type="cluster", cov_kwds={"groups": data[self.cluster_cov]})
        elif self.cov_type is not None:
            fin_reg_fit = fin_reg.fit(cov_type=self.cov_type)
        else:
            fin_reg_fit = fin_reg.fit()


        self.final_regression = fin_reg_fit
        self.selected_controls = selected_conts

        return fin_reg_fit
            
