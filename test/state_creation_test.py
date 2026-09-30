import polars as pl
import pytest
import numpy as np

from scripts.state_creation import create_state

@pytest.fixture
def mock_dataset():
    data = {
        'customer_id': ["C01", "C02", "C03", "C04"],  
        'age': [21, 28, 50, 19],
        'monthly_spend': [100.00, 150.69, 67.67, 19.19],
        'subscription_tier': ["Free", "Standard", "Free", "Premium"],
        'account_age_months': [12, 43, 5, 19],
        'churn': [0, 1, 0, 0]
    }

    df = pl.DataFrame(data)

    return df

@pytest.fixture
def mock_dict():
    df_dict = {
        "goal": "classification model to predict customer churn",
        "target": "churn",
        "constraints": {
            "interpretability": "medium to high",
            "training time": "low to medium",
            "available resources": "medium"
        }
    }

    return df_dict

def test_create_state(mock_dataset, mock_dict):
    state = create_state(mock_dataset, mock_dict)

    assert state is not None, "The state dictionary was not created (returned None)."
    assert isinstance(state, dict), f"Expected a dictionary, but got {type(state)}."

    print(state)

    print("\nGenerated State Dictionary:", state)