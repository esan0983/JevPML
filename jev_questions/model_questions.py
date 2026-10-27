import warnings

from scripts import ask_jev

def model(state: dict) -> dict:
    questions = {
        "linear_regression": {
            "type": "noul",
            "instructions": "Is linear regression a suitable model for the given columns, dataset properties, and constraints?"
        },
        "logistic_regression": {
            "type": "noul",
            "instructions": "Is logistic regression a suitable model for the given columns, dataset properties, and constraints?"
        },
        "knn": {
            "type": "noul",
            "instructions": "Is K-Nearest Neighbors a suitable model for the given columns, dataset properties, and constraints?"
        },
        "svm": {
            "type": "noul",
            "instructions": "Is Support Vector Machines a suitable model for the given columns, dataset properties, and constraints?"
        },
        "naive_bayes": {
            "type": "noul",
            "instructions": "Is Naive Bayes a suitable model for the given columns, dataset properties, and constraints?"
        },
        "decision_trees": {
            "type": "noul",
            "instructions": "Is Decision Trees a suitable model for the given columns, dataset properties, and constraints?"
        },
        "random_forest": {
            "type": "noul",
            "instructions": "Is Random Forest a suitable model for the given columns, dataset properties, and constraints?"
        },
        "xgboost": {
            "type": "noul",
            "instructions": "Is XGBoost a suitable model for the given columns, dataset properties, and constraints?"
        },
        "catboost": {
            "type": "noul",
            "instructions": "Is CatBoost a suitable model for the given columns, dataset properties, and constraints?"
        },
        "pytorch": {
            "type": "noul",
            "instructions": "Is a custom PyTorch Neural Network a suitable model for the given columns, dataset properties, and constraints?"
        },
    }

    response = ask_jev(state, questions)
    answers = response.answers
    return_dict = {}

    for answer in answers:
        return_dict[answer] = answers[answer].noul

    sorted_dict = dict(sorted(return_dict.items(), key=lambda item: item[1], reverse=True))
    sorted_dict = {k: v for k, v in sorted_dict.items() if v > 0.6}

    if len(sorted_dict) == 0:
            warnings.warn("There are no models suitable for this dataset.")

    sorted_dict_filtered = dict(list(sorted_dict.items())[:3]) if len(sorted_dict) >= 3 else sorted_dict

    return sorted_dict_filtered