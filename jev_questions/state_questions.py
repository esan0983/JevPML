import warnings

from scripts import ask_jev

def role(col_name: str, data_type: str, num_unique: int) -> str:
    state = {
        "column name": col_name,
        "data type": data_type,
        "no. of unique values": num_unique
    }

    question = {
        "role": {
            "type": "choice",
            "instructions": "Given the column name and data type, what role does this column play?",
            "criteria": {
                "identifier": {
                    "examples": "Customer ID, Event ID"
                },
                "categorical non-ordinal": {
                    "examples": "Department Name, College Major, Team, Season"
                },
                "categorical ordinal": {
                    "examples": "Subscription Tier, Education Level, Socioeconomic Status"
                },
                "numerical": {
                    "examples": "Age, Monthly Spend, Distance, Cost",
                    "non-examples": "Variables that can be expressed as numerical values but are not inherently numerical like Churn, Success, any boolean variable"
                }
            }
        }
    }

    response = ask_jev(state, question)
    role = response.answers["role"]
    assert role

    if role.confidence < 0.7:
        warnings.warn(f"Confidence for selecting the role of the feature {col_name} is low. Double-check and proceed with caution.")

    return role.choice

def column_usefulness(state: dict) -> dict:
    assert "columns" in state, "There is no columns key in your state dictionary."
    columns = state.get("columns")
    questions = {}

    for column in columns:
        questions[column] = {
            "type": "noul",
            "instructions": f"Is the {column} column of this dataset useful for machine learning?",
            "criteria": {
                "true": "The column will definitely help with the machine learning phase as it is contextually relevant. Examples: age, income, subscriptin tier, etc.",
                "false": "This column serves no purpose in the machine learning phase. Examples: ID, Name, User ID, etc"
            }
        }

    response = ask_jev(state, questions)
    answers = response.answers
    return_dict = {}

    for column in columns:
        return_dict[column] = answers[column].noul

    return return_dict

