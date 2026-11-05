import numpy as np
import polars as pl

from typing import List

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import LogitResults
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def run_box_tidwell(df: pl.DataFrame, col: str, target: str) -> LogitResults:
    col_min = df[col].cast(pl.Float64).min()
    if col_min <= 0:
        df = df.with_columns((pl.col(col) + 1 - col_min).alias(col))

    df_box = df.filter(pl.col(col) > 0).with_columns(
        (pl.col(col) * pl.col(col).log()).alias("col_ln_col")
    )

    box_tidwell_model = smf.logit(
        f"{target} ~ {col} + col_ln_col", 
        data=df_box.to_pandas()
    ).fit()

    return box_tidwell_model

def filter_high_vif(df: pl.DataFrame, features: List[str], threshold: float = 10.0) -> pl.DataFrame:
    X = df.select(features).with_columns(pl.lit(1.0).alias("const"))
    
    X_matrix = X.to_numpy()
    
    vifs = [
        variance_inflation_factor(X_matrix, i) 
        for i in range(len(features))  # Features are indices 0 to len(features)-1; const is last
    ]
    
    vif_data = pl.DataFrame({
        "Feature": features,
        "VIF": vifs
    })
    
    filtered_vif = vif_data.filter(pl.col("VIF") > threshold)
    
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