import warnings
import polars as pl

from scripts import ask_jev
from utils.preprocessing_utils import (
    run_box_tidwell,
    filter_high_vif,
    filter_high_cooks_distance
)

class LRPreprocessing:
    def __init__(self, df: pl.DataFrame, target: str, identifier: str, state: dict):
        self.df = df
        self.target = target
        self.identifier = identifier
        self.state = state

    # Assumptions

    def check_binary(self) -> None:
        unique = self.df.n_unique(subset=self.target)
        assert unique == 2, f"Target column {self.target} is not binary and has {self.unique} unique values."
        print("Logistic Regression Assumption 1 (Binary Target Variable) Passed!")

    def independence(self) -> None:
        if self.identifier:
            repeated_values = self.df.filter(pl.col(self.identifier).is_duplicated()).select(pl.col(self.identifier).unique())

            # Note: it is assumed that row duplicates have been removed.
            assert len(repeated_values) == 0, "Independence of observations was violated for the following identifier values: " + ", ".join(repeated_values)
        print("Logistic Regression Assumption 2 (Independent Observations) Passed!")

    def linearity_logit(self) -> None:
        violated_cols = []

        for col in self.df.columns:
            if col == self.target or col == self.identifier:
                continue

            if self.state.get("columns", {}).get(col, {}).get("role", "") == "numerical":
                model = run_box_tidwell(self.df.drop([self.identifier]), col, self.target)
                p_val = model.pvalues['col_ln_col']

                if p_val < 0.05:
                    violated_cols.append(col)

        assert len(violated_cols) == 0, "Linearity of logit assumption was violated by the following columns: " + ", ".join(violated_cols)
        print("Logistic Regression Assumption 3 (Linearity of Logit) Passed!")

    def multicollinearity(self) -> None:
        filtered_df = filter_high_vif(self.df.drop([self.identifier]), self.df.drop([self.target, self.identifier]).columns)
        assert filtered_df.is_empty, f"The following columns failed the VIF multicollinearity test: {self.df.to_string()}"

    def influential_outliers(self) -> None:
        filtered_df = filter_high_cooks_distance(self.df.drop([self.identifier]), self.target)
        assert filtered_df.is_empty, f"The following rows failed the influential outliers test: {self.df.to_string()}"
        print("Logistic Regression Assumption 4 (No Influential Outliers) Passed!")

    def sample_size(self) -> None:
        assert self.df.height >= (self.df.width - 1) * 10, f"The rows to features ratio is less than 10:1. Either add more data or remove some features."
        print("Logistic Regression Assumption 5 (Sample Size) Passed!")

    # Scikit-learn requirements. The rest of them will be done during cross-validation

    def solver(self, state) -> str:
        # Change the question to pick the top 3 solvers instead
        # Not used for now
        question = {
            "solver": {
                "type": "choice",
                "instructions": "Given the entire dataset, which sklearn Logistic Regression solver should we use?",
                "criteria": {
                "lbfgs": "For small to medium datasets. Quasi-Newton method; fast convergence; robust for general use.",
                "liblinear": "For small datasets. Coordinate descent; highly recommended for high-dimensional data.",
                "saga": "For very large datasets. Stochastic Average Gradient Augmented.",
                "sag": "For large datasets. Stochastic Average Gradient; faster than standard SGD for large datasets.",
                "newton-cg": "For small to medium datasets. Newton-Conjugate Gradient; expensive computationally per iteration.",
                "newton-cholesky": "For medium to large datasets with n_samples >> n_features. Extremely fast when the number of samples is much larger than features."
                }
            }
        }

        response = ask_jev(state, question)
        solver = response.answers["solver"]
        choice = solver.choice

        print(f"Choosing solver {choice}")

        if solver.confidence < 0.7:
            warnings.warn(f"Confidence for selecting the solver is low. Double-check and proceed with caution.")

        return choice

def encode_column(df: pl.DataFrame, column: str) -> str:
    unique_categories = df[column].unique()
    num_unique = df.n_unique(column)

    state = {
        "column name": column,
        "number of unique values": str(num_unique),
        "categories": ", ".join(str(cat) for cat in unique_categories.to_list()) if num_unique < 50 else "over 50 unique values"
    }

    question = {
        "encoding": {  
            "type": "choice",  
            "instructions": f"What type of encoding should be done to the feature {column}?",  
            "criteria": {  
                "One-Hot Encoding": "Replaces a single categorical column with multiple binary columns--one for each unique category",  
                "Ordinal Encoding": "Converts categorical data with a natural rank or order into a sequence of numeric integers",  
                "Label Encoding": "Turns categories into numbers without any specific rank or order",  
                "Mean Encoding": "Replaces each categorical value with the average value of the target variable for that specific category"
            }  
        }
    }

    response = ask_jev(state, question)
    encoding = response.answers["encoding"]
    if encoding.confidence < 0.7:
        warnings.warn("The encoding method was chosen with low confidence. Double-check if necessary.")

    print(f"The chosen encoding feature for {column} is {encoding.choice}")
    return encoding.choice

def determine_lowest_value(values: set, column: str) -> str:
    state = {
        "column_name": column
    }

    question = {
        "lowest": {
            "type": "choice",
            "instructions": f"Which of the values in the column {column} is the lowest?",
            "criteria": {

            }
        }
    }

    criteria = question["lowest"]["criteria"]

    for value in values:
        criteria[value] = ""

    response = ask_jev(state, question)
    lowest = response.answers["lowest"]

    if lowest.confidence < 0.7:
        warnings.warn("The following choice for the order was chosen with low confidence. Double-check if necessary.")

    print(f"The next value in the order is {lowest.choice}")
    return lowest.choice