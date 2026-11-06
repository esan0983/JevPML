# Had Claude fix the issue of Polars columns being transformed into nparrays, and thus can't be filtered

import polars as pl
import polars.selectors as cs
import optuna
import numpy as np
import warnings

from pprint import pprint

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

from jev_questions.state_questions import column_usefulness

# FIX: sklearn transformers need 2D input and return numpy arrays, so these helpers
# wrap a single polars column in/out and keep the DataFrame a DataFrame.
def fit_transform_column(df: pl.DataFrame, column: str, transformer) -> pl.DataFrame:
    values = transformer.fit_transform(df.select(column).to_numpy()).ravel()
    return df.with_columns(pl.Series(column, values))

def transform_column(df: pl.DataFrame, column: str, transformer) -> pl.DataFrame:
    values = transformer.transform(df.select(column).to_numpy()).ravel()
    return df.with_columns(pl.Series(column, values))

def evaluate_columns(state: dict, df: pl.DataFrame, column_blacklist: list) -> pl.DataFrame:
    # Checks every column's noul score and filters them out accordingly.
    noul_dict = column_usefulness(state)

    print(column_blacklist)
    print(df.columns)

    for column in df.columns:
        if any(s in column for s in column_blacklist):
            continue
        assert column in noul_dict, f"The column {column} is not in the current state."
        if noul_dict[column] < 0.5:
            df = df.drop([column])

    return df

def initial_cleaning(state: dict, df: pl.DataFrame) -> pl.DataFrame: # performs initial cleaning on the dataset such as duplicates, messy strings, etc
    # Remove duplicates
    df = df.unique()

    # Get the target variable
    target = state.get("properties", {}).get("target_variable", "None")
    assert target in df.columns, f"Target variable {target} is not in the list of columns"

    # Go through each column and check if it's categorical. If it is, clean strings
    columns = df.columns
    for column in columns:
        column_data = state.get("columns", {}).get(column, {})
        assert len(column_data) > 0

        if column_data.get("role", "") in ["categorical ordinal", "categorical non-ordinal"] and column != target:
            df = df.with_columns(
                pl.col(column).str.strip_chars().str.to_titlecase()
            )

            df = df.with_columns(
                pl.col(column).str.replace_all("-", "").alias(column)
            )

    return df

def check_assumptions(state: dict, df: pl.DataFrame, encoded_column_list: list) -> pl.DataFrame: # will stop the entire program if assumptions are not met
    print("Will now check LR assumptions.")
    if len(encoded_column_list) > 0:
        warnings.warn("The following columns will be encoded during the CV phase and thus will not be considered in the assumptions phase: " + ", ".join(encoded_column_list))
        df = df.drop(columns=encoded_column_list, errors="ignore")

    # Get the target variable
    target = state.get("properties", {}).get("target_variable", "None")
    assert target in df.columns, f"Target variable {target} is not in the list of columns"

    # Get identifier variable
    identifier = "None"
    columns = state["columns"]

    for column in columns:
        role = columns.get(column, {}).get("role", "None")
        if role == "identifier":
            identifier = column
            break

    assert identifier in df.columns, f"Identifier variable {identifier} is not in the list of columns"
        
    # Check each assumption
    assumptions = LRPreprocessing(df, target, identifier, state)
    assumptions.check_binary()
    assumptions.independence()
    # assumptions.linearity_logit() # commented for now, but will need a fix if it's violated
    # assumptions.multicollinearity() # commented for now due to divide-by-zero problems, most likely the result of the encoding stuff
    assumptions.influential_outliers() 
    assumptions.sample_size()
    
    return df

def assign_encoding(state: dict, df: pl.DataFrame) -> tuple[pl.DataFrame, list, dict, list]: # should return a list of columns that need to be encoded during CV
    assert "columns" in state, "There is no columns key in your state dictionary."
    columns = state.get("columns").copy()

    column_list = []
    column_blacklist = []

    for column in columns:
        needs_encoding = False
        if columns.get(column, {}).get("role", "None") == "identifier":
            continue

        is_numeric = df.schema[column].is_numeric()
        if not is_numeric:
            is_clean_numeric = (
                df.select(
                    (
                        pl.col(column).cast(pl.Float64, strict=False).is_not_null()
                        == pl.col(column).is_not_null()
                    ).all()
                ).item() 
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
                    encoded_matrix = encoder.fit_transform(df[[column]])

                    encoded_df = pl.DataFrame(
                        dict(zip(encoder.get_feature_names_out([column]), encoded_matrix.T))
                    )

                    df = df.drop(column).hstack(encoded_df)
                case "Label Encoding":
                    df = df.with_columns(
                        pl.col(column).cast(pl.Categorical).to_physical().alias(column)
                    )
                case "Ordinal Encoding":
                    order = []
                    values = set(df[column].drop_nulls().unique().to_list())
    
                    while len(values) > 0:
                        lowest = determine_lowest_value(values, column)
                        values.remove(lowest)
                        order.append(lowest)

                    ranking = [order]
                    encoder = OrdinalEncoder(categories=ranking)

                    X = df.select(column).to_numpy()
    
                    column_encoded = encoder.fit_transform(X)
                    encoded_list = column_encoded.flatten().tolist()
                    
                    df = df.with_columns(
                        pl.Series(name=column, values=encoded_list)
                    )
                case "Mean Encoding":
                    column_list.append(column)

            # Remove from state to bypass column usefulness check since they're useful by design
            if encoding_method == "One-Hot Encoding":
                temp = state["columns"].pop(column)
                column_blacklist.append(column)

    print(df.columns)

    return df, column_list, state, column_blacklist

def logistic_regression(df: pl.DataFrame, state: dict, column_encoding_list: list, column_blacklist: list) -> None:
    # Get the target variable
    target = state.get("properties", {}).get("target_variable", "None")
    assert target in df.columns, f"Target variable {target} is not in the list of columns"

    # Drop rows where target is NaN
    df = df.drop_nans(subset=[target])

    X = df.drop([target])
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
            # Replaces penalty: 0.0 is pure L2, 1.0 is pure L1, intermediate values are ElasticNet
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'solver': 'saga',  
            'max_iter': 2000,
            'random_state': 42,
        }

        # Perform stratified K-fold cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_losses = []  # FIX: was reset inside the fold loop, so only one fold was ever averaged

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_pretrain, y_pretrain)):
            print(f"Fold number {fold}")

            X_train, X_val = X_pretrain[train_idx], X_pretrain[val_idx]
            y_train, y_val = y_pretrain[train_idx], y_pretrain[val_idx]

            num_missing_rows = len(X_train.filter(pl.any_horizontal(pl.all().is_null())))
            total_rows = X_train.height

            columns = X_train.columns

            # Check if mean encoding needs to be done
            # FIX: TargetEncoder returned a numpy array (and was re-run once per matching column on the
            # whole frame). Now it runs once, only on the target-encoded columns, and returns polars.
            te_cols = [c for c in columns if c in column_encoding_list]
            if te_cols:
                encoder = TargetEncoder(smooth="auto").set_output(transform="polars")
                X_train = X_train.with_columns(encoder.fit_transform(X_train.select(te_cols), y_train))
                X_val = X_val.with_columns(encoder.transform(X_val.select(te_cols)))

            # Imputation
            if num_missing_rows / total_rows > 0.25:
                # Multiple Imputation
                imputer = IterativeImputer(
                    sample_posterior=False,
                    max_iter=10,
                    random_state=42
                )
                imputer.set_output(transform="polars")  # FIX: keep polars DataFrames

                X_train = imputer.fit_transform(X_train)
                X_val = imputer.transform(X_val)
            else:
                # Column-by-column imputation
                for column in columns:
                    column_data = {}
                    if not any(s in column for s in column_blacklist):
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

                        outlier_count = len(X_train.filter(
                            (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                        ))

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
                            X_train = fit_transform_column(X_train, column, imputer)
                            X_val = transform_column(X_val, column, imputer)
                        case "Median":
                            imputer = SimpleImputer(strategy='median')
                            X_train = fit_transform_column(X_train, column, imputer)
                            X_val = transform_column(X_val, column, imputer)
                        case "Mode":
                            imputer = SimpleImputer(strategy='most_frequent')
                            X_train = fit_transform_column(X_train, column, imputer)
                            X_val = transform_column(X_val, column, imputer)
                        case "Constant":
                            imputer = SimpleImputer(strategy='constant', fill_value=0)
                            X_train = fit_transform_column(X_train, column, imputer)
                            X_val = transform_column(X_val, column, imputer)

            # Scaling
            for column in columns:
                column_data = {}
                if not any(s in column for s in column_blacklist):
                    column_data = state.get("columns", {}).get(column, {})

                if column_data.get("role", "") == "numerical":
                    q25 = pl.col(column).quantile(0.25)
                    q75 = pl.col(column).quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    outlier_count = X_train.filter(
                        (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                    ).height

                    if outlier_count / total_rows > 0.05:
                        scaler = RobustScaler(unit_variance=True)
                    else:
                        scaler = StandardScaler()

                    X_train = fit_transform_column(X_train, column, scaler)
                    X_val = transform_column(X_val, column, scaler)

            model = LogisticRegression(**params, class_weight='balanced')
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
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
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
    num_missing_rows = X_pretrain.filter(pl.any_horizontal(pl.all().is_null())).height
    total_rows = X_pretrain.height
    columns = X_pretrain.columns

    # Check if mean encoding needs to be done
    # FIX: same TargetEncoder fix as in the CV loop
    te_cols = [c for c in columns if c in column_encoding_list]
    if te_cols:
        encoder = TargetEncoder(smooth="auto").set_output(transform="polars")
        X_pretrain = X_pretrain.with_columns(encoder.fit_transform(X_pretrain.select(te_cols), y_pretrain))
        X_test = X_test.with_columns(encoder.transform(X_test.select(te_cols)))

    if num_missing_rows / total_rows > 0.25:
        # Multiple Imputation
        imputer = IterativeImputer(
            sample_posterior=False,
            max_iter=10,
            random_state=42
        )
        imputer.set_output(transform="polars")

        # FIX: these used the stale CV variables X_train / X_val
        X_pretrain = imputer.fit_transform(X_pretrain)
        X_test = imputer.transform(X_test)
    else:
        # Column-by-column imputation
        for column in columns:
            column_data = {}
            if not any(s in column for s in column_blacklist):
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
                ).height

                if mi_score > 0.01:
                    imputation_method = "Constant"
                elif abs(skew(X_pretrain[column], bias=True)) > 1 or outlier_count / total_rows > 0.01:  # FIX: was X_train
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
                    X_pretrain = fit_transform_column(X_pretrain, column, imputer)
                    X_test = transform_column(X_test, column, imputer)
                case "Median":
                    imputer = SimpleImputer(strategy='median')
                    X_pretrain = fit_transform_column(X_pretrain, column, imputer)
                    X_test = transform_column(X_test, column, imputer)
                case "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                    X_pretrain = fit_transform_column(X_pretrain, column, imputer)
                    X_test = transform_column(X_test, column, imputer)
                case "Constant":
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                    X_pretrain = fit_transform_column(X_pretrain, column, imputer)
                    X_test = transform_column(X_test, column, imputer)

    for column in columns:
        column_data = {}
        if not any(s in column for s in column_blacklist):
            column_data = state.get("columns", {}).get(column, {})
            assert len(column_data) > 0

        if column_data.get("role", "") == "numerical":
            # FIX: q25/q75/iqr were stale leftovers from a previous loop; recompute per column
            q25 = pl.col(column).quantile(0.25)
            q75 = pl.col(column).quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outlier_count = X_pretrain.filter(
                (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
            ).height

            if outlier_count / total_rows > 0.05:
                scaler = RobustScaler(unit_variance=True)
            else:
                scaler = StandardScaler()

            X_pretrain = fit_transform_column(X_pretrain, column, scaler)
            X_test = transform_column(X_test, column, scaler)

    final_model = LogisticRegression(**study.best_params, solver='saga', class_weight='balanced')
    final_model.fit(X_pretrain, y_pretrain)
    test_preds = final_model.predict(X_test)
    test_probs = final_model.predict_proba(X_test)[:, 1]  # FIX: predict_proba is a method

    print("Testing phase finished!")
    print(classification_report(y_test, test_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, test_probs):.4f}")

def lr_pipeline(state: dict, df: pl.DataFrame) -> None:
    df = initial_cleaning(state, df)
    df, encoded_column_list, state, column_blacklist = assign_encoding(state, df)
    df = check_assumptions(state, df, encoded_column_list)
    df = evaluate_columns(state, df, column_blacklist)
    logistic_regression(df, state, encoded_column_list, column_blacklist)