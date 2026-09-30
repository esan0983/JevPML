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
