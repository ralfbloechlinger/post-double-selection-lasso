import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class PDSLasso:
    def __init__(self, data: pd.DataFrame, y: str, d: str, control_cols: list[str] | None = None):
        self.data = data
        self.y = y 
        self.d = d
        self.control_cols = control_cols

    def __repr__(self):
        repr_str = f"PDS-Lasso class with dependent variable {self.y}, variable of interest {self.d}"
        return repr_str
    
    def prep_data(self):
         if self.control_cols is None:
            return (self.data[self.y], self.data[self.d])
         else:
            return (self.data[self.y], self.data[self.d], self.data[self.control_cols])
         


    def _run_lasso(self, X_ctrl, y_vec):
        X_ctrl_std = StandardScaler().fit_transform(X_ctrl)
        lasso_fit = LassoCV().fit(X = X_ctrl_std, y = y_vec)
        coefs = lasso_fit.coef_
        selected = [col for col, c in zip(self.control_cols, coefs) if c != 0]
        return lasso_fit, selected
    
    def fit(self):

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
            
