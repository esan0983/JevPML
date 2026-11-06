import numpy as np
import polars as pl
import polars.selectors as cs
import warnings

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

def filter_high_vif(df: pl.DataFrame, features: list[str], threshold: float = 10.0) -> pl.DataFrame:
    df_subset = df.select(features)
    clean_df = df_subset.filter(
        ~pl.any_horizontal(
            cs.numeric().is_null() | cs.numeric().is_nan() | cs.numeric().is_infinite()
        )
    )
    
    dropped_count = df.height - clean_df.height
    if dropped_count > 0:
        warnings.warn(f"Dropped {dropped_count} rows ({dropped_count / df.height * 100:.2f}%) due to invalid values.")

    if clean_df.height == 0:
        raise ValueError("No rows remaining after cleaning.")

    X = clean_df.with_columns(pl.lit(1.0).alias("const"))
    X_matrix = X.to_numpy()
    
    vifs = [
        variance_inflation_factor(X_matrix, i) 
        for i in range(len(features)) 
    ]
    
    vif_data = pl.DataFrame({
        "Feature": features,
        "VIF": vifs
    })
    
    return vif_data.filter(pl.col("VIF") > threshold)

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
    
    return df.gather(influential_indices)