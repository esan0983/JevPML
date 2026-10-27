import polars as pl

from jev_questions.state_questions import column_usefulness

def evaluate_columns(state: dict, df: pl.DataFrame) -> pl.DataFrame:
    # Checks every column's noul score and filters them out accordingly.
    noul_dict = column_usefulness(state)

    for column in df.columns:
        assert column in noul_dict, f"The column {column} is not in the current state."
        if noul_dict[column] < 0.5:
            df = df.drop(columns=[column], errors='ignore')

    return df