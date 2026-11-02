# This assumes that the dataset has gone through "column usefulness" tests and thus all the useless columns are dropped.

import polars as pl
import optuna
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
    TargetEncoder
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.experimental import (
    enable_iterative_imputer
)

from sklearn.impute import (
    SimpleImputer,
    IterativeImputer
)

from sklearn.feature_selection import (
    mutual_info_classif
)

from scipy.stats import (
    skew
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    log_loss,
    classification_report,
    roc_auc_score
)

from jev_questions.ml_algorithms.logistic_questions import (
    encode_column,
    determine_lowest_value,
    LRPreprocessing
)

def initial_cleaning(state: dict, df: pl.DataFrame) -> pl.DataFrame: # performs initial cleaning on the dataset such as duplicates, messy strings, etc
    # Remove duplicates
    df = df.unique()

    # Go through each column and check if it's categorical. If it is, clean strings
    columns = df.columns
    for column in columns:
        column_data = state.get("columns", {}).get(column, {})
        assert len(column_data) > 0

        if column_data.get("role", "") in ["categorical ordinal", "categorical non-ordinal"]:
            df = df.with_columns(
                pl.col(column).str.strip_chars().str.to_titlecase()
            )

def check_assumptions(state: dict, df: pl.DataFrame) -> None: # will stop the entire program if assumptions are not met
    # Will complete
    return None

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

def logistic_regression(df: pl.DataFrame, target: str, state: dict, column_encoding_list: list) -> None:
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

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga',  # 'saga' supports both l1 and l2 penalties
            'max_iter': 2000,
            'random_state': 42
        }

        # Perform stratified K-fold cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_pretrain)):
            print(f"Fold number {fold}")
            fold_losses = []

            X_train, X_val = X_pretrain[train_idx], X_pretrain[val_idx]
            y_train, y_val = y_pretrain[train_idx], y_pretrain[val_idx]

            num_missing_rows = X_train.filter(pl.any_horizontal(pl.all().is_null())).len()
            total_rows = X_train.height

            columns = X_train.columns

            # Check if mean encoding needs to be done
            for column in columns:
                if column in column_encoding_list:
                    encoder = TargetEncoder(smooth="auto")
                    X_train = encoder.fit_transform(X_train, y_train)
                    X_val = encoder.transform(X_val)

            # Imputation
            if num_missing_rows / total_rows > 0.25:
                # Multiple Imputation
                imputer = IterativeImputer(
                    sample_posterior=False,
                    max_iter=10,
                    random_state=42
                )

                X_train = imputer.fit_transform(X_train)
                X_val = imputer.transform(X_val)
            else:
                # Column-by-column imputation
                for column in columns:
                    column_data = state.get("columns", {}).get(column, {})
                    assert len(column_data) > 0

                    imputation_method = "Drop"

                    # No need for Jev for automatically choosing a simple imputation method.
                    if column_data.get("role", "") in ["categorical ordinal", "categorical non-ordinal"]:
                        imputation_method = "Mode"
                    else:
                        is_missing = X_train[column].is_null().cast(pl.Int8)
                        X_missing_matrix = is_missing.to_numpy().reshape(-1, 1)
                        mi_score = mutual_info_classif(X_missing_matrix, y_train.to_numpy(), random_state=42)[0]

                        q25 = pl.col(column).quantile(0.25)
                        q75 = pl.col(column).quantile(0.75)
                        iqr = q75 - q25

                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr

                        outlier_count = X_train.filter(
                            (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                        ).len()

                        if mi_score > 0.01:
                            imputation_method = "Constant"
                        elif abs(skew(X_train[column], bias=True)) > 1 or outlier_count / total_rows > 0.01:
                            imputation_method = "Median"
                        else:
                            imputation_method = "Mean"
                            
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
                        case "Mean":
                            imputer = SimpleImputer(strategy='mean')
                            X_train[column] = imputer.fit_transform(X_train[column])
                            X_val[column] = imputer.transform(X_val[column])
                        case "Median":
                            imputer = SimpleImputer(strategy='median')
                            X_train[column] = imputer.fit_transform(X_train[column])
                            X_val[column] = imputer.transform(X_val[column])
                        case "Mode":
                            imputer = SimpleImputer(strategy='most_frequent')
                            X_train[column] = imputer.fit_transform(X_train[column])
                            X_val[column] = imputer.transform(X_val[column])
                        case "Constant":
                            imputer = SimpleImputer(strategy='constant', fill_value=0)
                            X_train[column] = imputer.fit_transform(X_train[column])
                            X_val[column] = imputer.transform(X_val[column])

            # Scaling
            for column in columns:
                column_data = state.get("columns", {}).get(column, {})
                assert len(column_data) > 0

                if column_data.get("role", "") == "numerical":
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    outlier_count = X_train.filter(
                        (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                    ).len()

                    if outlier_count / total_rows > 0.05:
                        scaler = RobustScaler(unit_variance=True)
                    else:
                        scaler = StandardScaler()

                    X_train[column] = scaler.fit_transform(X_train[column])
                    X_val[column] = scaler.transform(X_val[column])

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)
            loss = log_loss(y_val, y_pred_proba)
            fold_losses.append(loss)

            trial.report(np.mean(fold_losses), fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_losses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.TPESampler(n_startup_trials=10, seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0
        )
    )
    study.optimize(objective, n_trials=50, n_jobs=1)

    print("Best parameters found:")
    print(study.best_params)
    print(f"Best CV log loss: {study.best_value:.4f}")
    print("Moving to the testing phase...")

    # Train-test imputation and scaling
    # Both imputation and scaling are just duplicates of what we did above. We can probably compress this by making a function and just
    # calling it twice, but the variable names would have to be changed. Not a high priority.
    num_missing_rows = X_pretrain.filter(pl.any_horizontal(pl.all().is_null())).len()
    total_rows = X_pretrain.height
    columns = X_pretrain.columns

    # Check if mean encoding needs to be done
    for column in columns:
        if column in column_encoding_list:
            encoder = TargetEncoder(smooth="auto")
            X_pretrain = encoder.fit_transform(X_pretrain, y_pretrain)
            X_test = encoder.transform(X_test)

    if num_missing_rows / total_rows > 0.25:
        # Multiple Imputation
        imputer = IterativeImputer(
            sample_posterior=False,
            max_iter=10,
            random_state=42
        )

        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
    else:
        # Column-by-column imputation
        for column in columns:
            column_data = state.get("columns", {}).get(column, {})
            assert len(column_data) > 0

            imputation_method = "Drop"

            # No need for Jev for automatically choosing a simple imputation method.
            if column_data.get("role", "") in ["categorical ordinal", "categorical non-ordinal"]:
                imputation_method = "Mode"
            else:
                is_missing = X_pretrain[column].is_null().cast(pl.Int8)
                X_missing_matrix = is_missing.to_numpy().reshape(-1, 1)
                mi_score = mutual_info_classif(X_missing_matrix, y_pretrain.to_numpy(), random_state=42)[0]

                q25 = pl.col(column).quantile(0.25)
                q75 = pl.col(column).quantile(0.75)
                iqr = q75 - q25

                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr

                outlier_count = X_pretrain.filter(
                    (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                ).len()

                if mi_score > 0.01:
                    imputation_method = "Constant"
                elif abs(skew(X_train[column], bias=True)) > 1 or outlier_count / total_rows > 0.01:
                    imputation_method = "Median"
                else:
                    imputation_method = "Mean"
                    
            match imputation_method:
                case "Drop":
                    null_list_pretrain = X_pretrain.select(pl.col(column).is_null().arg_true()).item().to_list()
                    null_list_test = X_test.select(pl.col(column).is_null().arg_true()).item().to_list()

                    X_pretrain = X_pretrain.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_pretrain)
                    )

                    y_pretrain = y_pretrain.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_pretrain)
                    )

                    X_test = X_test.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_test)
                    )

                    y_test = y_test.filter(
                        ~pl.arange(0, pl.count()).is_in(null_list_test)
                    )
                case "Mean":
                    imputer = SimpleImputer(strategy='mean')
                    X_pretrain[column] = imputer.fit_transform(X_pretrain[column])
                    X_test[column] = imputer.transform(X_test[column])
                case "Median":
                    imputer = SimpleImputer(strategy='median')
                    X_pretrain[column] = imputer.fit_transform(X_pretrain[column])
                    X_test[column] = imputer.transform(X_test[column])
                case "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                    X_pretrain[column] = imputer.fit_transform(X_pretrain[column])
                    X_test[column] = imputer.transform(X_test[column])
                case "Constant":
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                    X_pretrain[column] = imputer.fit_transform(X_pretrain[column])
                    X_test[column] = imputer.transform(X_test[column])

    for column in columns:
        column_data = state.get("columns", {}).get(column, {})
        assert len(column_data) > 0

        if column_data.get("role", "") == "numerical":
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outlier_count = X_pretrain.filter(
                (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
            ).len()

            if outlier_count / total_rows > 0.05:
                scaler = RobustScaler(unit_variance=True)
            else:
                scaler = StandardScaler()

            X_pretrain[column] = scaler.fit_transform(X_pretrain[column])
            X_test[column] = scaler.transform(X_test[column])

    final_model = LogisticRegression(**study.best_params)
    final_model.fit(X_pretrain, y_pretrain)
    test_preds = final_model.predict(X_test)
    test_probs = final_model.predict_proba[:, 1]

    print("Testing phase finished!")
    print(classification_report(y_test, test_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, test_probs):.4f}")