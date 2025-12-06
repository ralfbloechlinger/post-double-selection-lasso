
import pandas as pd
import numpy as np

def simulate_pds_data(n=1000, p=100, true_effect=3.0, random_seed=42):
    """
    Simulates data for Post-Double Selection Lasso.
    
    Args:
        n: Sample size
        p: Total number of control variables (sparsity implies most are irrelevant)
        true_effect: The actual causal effect of d on y
        random_seed: For reproducibility
        
    Returns:
        df: Pandas DataFrame containing y, d, and X controls
        support_indices: Dictionary indicating which variables are truly relevant
    """
    np.random.seed(random_seed)
    
    # 1. Generate High-Dimensional Controls (X)
    # We use independent normal, but in reality, they might be correlated.
    X = np.random.normal(0, 1, size=(n, p))
    
    # 2. Define Sparsity (True Coefficients)
    # We will pick 3 sets of variables to be "active"
    
    # Set A: Confounders (Affect both D and Y) - These create omitted variable bias
    confounder_idxs = [0, 1, 2]
    
    # Set B: Instruments/Treatment Predictors (Affect D only)
    # If we miss these in selection, we lose efficiency or bias control
    treat_pred_idxs = [3, 4, 5]
    
    # Set C: Outcome Predictors (Affect Y only)
    outcome_pred_idxs = [6, 7, 8]
    
    # All other variables (9 to p) are pure noise
    
    # 3. Generate Treatment (d)
    # d depends on Confounders + Treatment Predictors
    # We create a continuous latent variable then binarize it for an "indicator"
    gamma = np.zeros(p)
    gamma[confounder_idxs] = [1.5, -1.0, 0.8]  # Confounder weights
    gamma[treat_pred_idxs] = [1.0, 0.5, 1.0]   # Treatment predictor weights
    
    d_latent = X @ gamma + np.random.normal(0, 1, n)
    d = (d_latent > 0).astype(int) # Binary Treatment Indicator
    
    # 4. Generate Outcome (y)
    # y depends on Treatment + Confounders + Outcome Predictors
    beta = np.zeros(p)
    beta[confounder_idxs] = [2.0, -1.5, 1.2]   # Confounder weights (correlated with d!)
    beta[outcome_pred_idxs] = [1.0, 1.0, 1.0]  # Outcome predictor weights
    
    # y = treatment_effect * d + (controls * beta) + noise
    y = true_effect * d + X @ beta + np.random.normal(0, 2, n)
    
    # Create DataFrame
    col_names = [f'x{i}' for i in range(p)]
    df = pd.DataFrame(X, columns=col_names)
    df['d'] = d
    df['y'] = y
    
    metadata = {
        'confounders': [f'x{i}' for i in confounder_idxs],
        'treatment_controls': [f'x{i}' for i in treat_pred_idxs],
        'outcome_controls': [f'x{i}' for i in outcome_pred_idxs]
    }
    
    return df, metadata