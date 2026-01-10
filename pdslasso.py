import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class PDSLasso:
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
        self.data = data
        self.y = y 
        self.d = d
        self.control_cols = control_cols
        self.lasso_penalty = lasso_penalty_cv
        self.penalty_c = penalty_c
        self.penalty_gamma = penalty_gamma
        self.penalty_sigma = penalty_sigma

    def __repr__(self):
        repr_str = f"PDS-Lasso class with dependent variable {self.y}, variable of interest {self.d}"
        return repr_str
    
    def prep_data(self):
         if self.control_cols is None:
            return (self.data[self.y], self.data[self.d])
         else:
            return (self.data[self.y], self.data[self.d], self.data[self.control_cols])
         

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
    
    def fit(self):

        # no controls => simple OLS
        if self.control_cols is None:
            y_vec, d_vec = self.prep_data()
            fin_reg = sm.OLS(y_vec, d_vec)
            selected_conts = None
            selected_vars = [self.d]

        else:
            y_vec, d_vec, X_control = self.prep_data()

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
            
