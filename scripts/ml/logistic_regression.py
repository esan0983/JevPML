# This assumes that the dataset has gone through "column usefulness" tests and thus all the useless columns are dropped.

import polars as pl

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.impute import (
    SimpleImputer
)

from jev_questions.ml_algorithms.logistic_questions import (
    encode_column,
    determine_lowest_value,
    impute_column
)

def assign_encoding(state: dict, df: pl.DataFrame) -> tuple[pl.DataFrame, list]: # should return a list of columns that need to be encoded during CV
    assert "columns" in state, "There is no columns key in your state dictionary."
    columns = state.get("columns")

    needs_encoding = False

    column_list = []

    for column in columns:
        is_numeric = df.schema[column].is_numeric()
        if not is_numeric:
            is_clean_numeric = (
                df.select(
                    pl.col(column).cast(pl.Float64, strict=False).is_not_null()
                    == pl.col(column).is_not_null()
                ).all().item() 
            )

            if is_clean_numeric:
                df = df.with_columns(
                    pl.col(column).cast(pl.Float64)
                )
            else:
                needs_encoding = True

        if needs_encoding:
            encoding_method = encode_column(df, column)
            match encoding_method:
                case "One-Hot Encoding":
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_matrix = encoder.fit_transform(df[column])

                    encoded_df = pl.DataFrame(
                        dict(zip(encoder.get_feature_names_out(['City']), encoded_matrix.T))
                    )

                    df = df.drop(column).hstack(encoded_df)
                case "Label Encoding":
                    df_encoded = df.with_columns(
                        pl.col(column).cast(pl.Categorical).to_physical().alias("city_encoded")
                    )
                case "Ordinal Encoding":
                    order = []
                    values = set(df[column].unique())
                    
                    while len(values) > 1:
                        lowest = determine_lowest_value(values, column)
                        values.remove(lowest)
                        order.append(lowest)

                    ranking = [order]
                    encoder = OrdinalEncoder(categories=ranking)
                    column_encoded = encoder.fit_transform(df[column])
                    df[column] = column_encoded
                case "Mean Encoding":
                    column_list.append(column)

    return df, column_list

def logistic_regression(df: pl.DataFrame, target: str, state: dict) -> None:
    # Drop rows where target is NaN
    df = df.drop_nans(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # Split into pretrain and test
    X_pretrain, X_test, y_pretrain, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    # Perform stratified K-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pretrain)):
        X_train, X_val = X_pretrain[train_idx], X_pretrain[val_idx]
        y_train, y_val = y_pretrain[train_idx], y_pretrain[val_idx]

        # CV imputation + scaling, iterated through columns
        columns = X_train.columns
        for column in columns:
            column_state = {
                "all columns": ", ".join(col for col in columns),
                "column of interest": column,
                "number of rows": str(X_train.height),
                "number of nulls": str(X_train[column].null_count())
            }

            imputation_method = impute_column(column_state)

            # Still need an option where a lot of data is missing and thus MICE should be used, 10% is a good threshold
            match imputation_method:
                case "Drop":
                    null_list_train = X_train.select(pl.col(column).is_null().arg_true()).item().to_list()
                    null_list_val = X_val.select(pl.col(column).is_null().arg_true()).item().to_list()

                    X_train = X_train.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_train)
                    )

                    y_train = y_train.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_train)
                    )

                    X_val = X_val.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_val)
                    )

                    y_val = y_val.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_val)
                    )
                case "Mean Imputation":
                    imputer = SimpleImputer(strategy='mean')
                    X_train[column] = imputer.fit_transform(X_train[column])
                    X_val[column] = imputer.transform(X_val[column])
                    

