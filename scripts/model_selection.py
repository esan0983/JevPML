import polars as pl

from jev_questions.model_questions import model

def select_model(state: dict) -> dict:
    return model(state)