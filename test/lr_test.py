import polars as pl
from pprint import pprint

from scripts.state_creation import create_state
from scripts.model_selection import select_model
from scripts.ml.logistic_regression import lr_pipeline

df = pl.read_csv("data/churn_bank.csv")

df_dict = {
    "goal": "classification model to predict customer churn",
    "target": "churned",
    "constraints": {
        "interpretability": "medium to high",
        "training time": "low to medium",
        "available resources": "medium"
    }
}

state = create_state(df, df_dict)
    
assert state is not None, "The state dictionary was not created (returned None)."
assert isinstance(state, dict), f"Expected a dictionary, but got {type(state)}."

print("Generated State Dictionary:")
pprint(state, indent=4)
lr_pipeline(state, df)