import polars as pl

from jev_questions.state_questions import role

def create_state(df: pl.DataFrame, df_dict: dict) -> dict:
    col_list = df.columns
    assert "target" in df_dict
    assert "goal" in df_dict
    assert "constraints" in df_dict
    
    state = {
        "goal": df_dict.get("goal", ""),
        "properties": {
            "no. of rows": df.height, 
            "target_variable": df_dict.get("target", "")
        },
        "columns": {},
        "constraints": df_dict.get("constraints", "")
    }

    col_dict = state["columns"]

    for col in col_list:
        dtype = str(df.schema[col])
        unique = df.n_unique(subset=col)

        col_dict[col] = {
            "data type": dtype,
            "nulls": str(df[col].null_count()),
            "role": role(col, dtype, unique),
            # Scalability will come from "role." Specifically, if it's inherently numerical, we can scale it.
        }

    # IDEA: check if data type makes sense. For example: if it's numeric but the dtype is string, get rid of the non-numeric ones and turn it into numeric
    # If it's categorical, check if there's a numeric value.
    # Use speculative fan-out

    return state