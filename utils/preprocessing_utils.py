import numpy as np
import polars as pl

from typing import List

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import LogitResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def run_box_tidwell(df: pl.DataFrame, col: str) -> LogitResultsWrapper:
    col_min = df[col].min()
    if col_min <= 0:
        df[col] += 1 - col_min

    df_box = df[df[col] > 0].copy()
    df_box['col_ln_col'] = df_box[col] * np.log(df_box[col])

    box_tidwell_model = smf.logit(f"outcome ~ {col} + col_ln_col", data=df_box).fit()

    return box_tidwell_model

def filter_high_vif(df: pl.DataFrame, features: List[str], threshold: float = 10.0) -> pl.DataFrame:
    X = df[features].copy()
    X = add_constant(X)

    vif_data = pl.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) 
        for i in range(1, X.shape[1])
    ]

    filtered_vif = vif_data[vif_data["VIF"] > threshold].reset_index(drop=True)
    
    return filtered_vif

def filter_high_cooks_distance(df: pl.DataFrame, target: str) -> pl.DataFrame:
    predictors = [col for col in df.columns if col != target]
    
    formula = f"{target} ~ {' + '.join(predictors)}"
    
    model = smf.logit(formula, data=df).fit(disp=0)
    
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    n = len(df)
    k = len(predictors)
    threshold = 4 / (n - k - 1)
    
    influential_indices = np.where(cooks_d > threshold)[0]
    
    print(f"Total rows: {n}")
    print(f"Threshold (4/n): {threshold:.4f}")
    print(f"Found {len(influential_indices)} influential observation(s).")
    
    return df.iloc[influential_indices]