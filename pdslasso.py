"""Post-double-selection Lasso estimator for a single treatment effect.

Implements the procedure described in Belloni, Chernozhukov, and Hansen (2014)
to select controls via two Lasso fits and then estimate the treatment effect
with OLS using heteroskedasticity-robust (HC1) standard errors.

Algorithm overview:
1) Lasso: d on X to select controls predictive of treatment.
2) Lasso: y on X to select controls predictive of outcome.
3) OLS: y on d and the union of selected controls.

Example:
    from pdslasso import PDSLasso
    est = PDSLasso(data=df, y="y", d="d", control_cols=["x0", "x1"])
    res = est.fit()
"""

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
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
        lasso_penalty_cv: bool = False,
        penalty_c: float = 1.1,
        penalty_gamma: float = 0.05,
        penalty_sigma: float | None = None,
    ):
        """Initialize the estimator with data columns and penalty settings.

        Args:
            data: Input DataFrame containing outcome, treatment, and controls.
            y: Column name for the outcome variable.
            d: Column name for the treatment variable.
            control_cols: Column names for candidate controls. If None, fit OLS
                with only the treatment variable.
            lasso_penalty_cv: If True, use cross-validation to select the Lasso
                penalty; otherwise use the parametric penalty from Belloni et al.
            penalty_c: Positive constant for the parametric penalty formula.
            penalty_gamma: Value in (0, 1) for the parametric penalty formula.
            penalty_sigma: Optional fixed sigma for the parametric penalty. If
                None, sigma is estimated from the response.

        Raises:
            KeyError: If any of y, d, or control_cols are missing from data.
        """
        self.data = data
        self.y = y 
        self.d = d
        self.control_cols = control_cols
        self.lasso_penalty = lasso_penalty_cv
        self.penalty_c = penalty_c
        self.penalty_gamma = penalty_gamma
        self.penalty_sigma = penalty_sigma

    def __repr__(self) -> str:
        repr_str = f"PDS-Lasso class with dependent variable {self.y}, variable of interest {self.d}"
        return repr_str
    
    def prep_data(self) -> PDSData:
        """Prepare and return the response, treatment, and optional controls."""
        y_vec = self.data[self.y]
        d_vec = self.data[self.d]
        X_ctrl = None if self.control_cols is None else self.data[self.control_cols]
        return PDSData(y=y_vec, d=d_vec, X=X_ctrl)
         

    def _estimate_sigma(self, y_vec: np.ndarray | pd.Series) -> float:
        if self.penalty_sigma is not None:
            return float(self.penalty_sigma)
        return float(np.std(y_vec, ddof=1))


    def _run_lasso(self, X_ctrl: np.ndarray | pd.DataFrame, y_vec: np.ndarray | pd.Series):

        # standardise controls for lasso
        X_ctrl_std = StandardScaler().fit_transform(X_ctrl)

        # use CV for choice of penalty level lambda
        if self.lasso_penalty:
            lasso_fit = LassoCV().fit(X=X_ctrl_std, y=y_vec)
        # use parametric penalty level from Belloni et al. (2014)
        else:
            n_obs, n_ctrl = X_ctrl_std.shape
            if not 0 < self.penalty_gamma < 1:
                raise ValueError("penalty_gamma must be between 0 and 1.")
            if self.penalty_c <= 0:
                raise ValueError("penalty_c must be positive.")
            sigma_hat = self._estimate_sigma(y_vec)
            if sigma_hat <= 0:
                raise ValueError("Estimated sigma must be positive.")
            # Parametric penalty level from Belloni et al. (2014), eq. (2.11).
            penalty_level = 2 * self.penalty_c * sigma_hat * math.sqrt(
                2 * n_obs * math.log(2 * n_ctrl / self.penalty_gamma)
            )
            alpha = penalty_level / (2 * n_obs)
            lasso_fit = Lasso(alpha=alpha, max_iter=10000).fit(X=X_ctrl_std, y=y_vec)

        # extract coefficients and selected controls
        coefs = lasso_fit.coef_
        selected = [col for col, c in zip(self.control_cols, coefs) if c != 0]

        return lasso_fit, selected
    
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

        # no controls => simple OLS
        if self.control_cols is None:
            selected_conts = None
            selected_vars = [self.d]

        else:
            X_control = data.X

            # Lasso 1: treatment indicator on all other controls
            lasso_1, selected_cols_d_on_X = self._run_lasso(X_ctrl=X_control, y_vec=d_vec)
            # Lasso 2: outcome on all other controls
            lasso_2, selected_cols_y_on_X = self._run_lasso(X_ctrl=X_control, y_vec=y_vec)
            
            self.first_stage_lasso = lasso_1
            self.second_stage_lasso = lasso_2 

            # selected controls as union of both sets of controls
            selected_conts = list(set(selected_cols_d_on_X + selected_cols_y_on_X))
            selected_vars = [self.d] + selected_conts

        # final matrix of X: variable of interest plus selected contrs
        X_final_vec = sm.add_constant(self.data[selected_vars])

        # fit object
        fin_reg = sm.OLS(y_vec, X_final_vec)
        fin_reg_fit = fin_reg.fit(cov_type="HC1")


        self.final_regression = fin_reg_fit
        self.selected_controls = selected_conts

        return fin_reg_fit
            
