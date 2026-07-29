import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Dependencies
    """)
    return


@app.cell
def _():
    import marimo as mo

    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from functools import partial
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    from sklearn.linear_model import LogisticRegression, SGDClassifier 
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

    return (
        ColumnTransformer,
        DecisionTreeClassifier,
        LGBMClassifier,
        LabelEncoder,
        LogisticRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RobustScaler,
        SGDClassifier,
        SimpleImputer,
        StratifiedKFold,
        XGBClassifier,
        balanced_accuracy_score,
        clone,
        compute_sample_weight,
        f1_score,
        mo,
        np,
        partial,
        pd,
        permutation_importance,
        sys,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Data
    """)
    return


@app.cell
def _(pd):
    main_dir = "playground-series-s6e7"

    train_data = pd.read_csv(f"{main_dir}/train.csv")
    test_data = pd.read_csv(f"{main_dir}/test.csv")

    train_data.shape, test_data.shape
    return main_dir, test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocess Data
    """)
    return


@app.cell
def _(ColumnTransformer, LabelEncoder, np, pd, sys, train_test_split):
    class Preprocessor:
        def __init__(
            self,
            data: pd.DataFrame,
            target: str = None,
            drop_id: bool = True,
            drop_nan: bool = False,
            drop_duplicates: bool = False,
        ):
            """Preprocessor class for cleaning, splitting, and transforming datasets.

            Supports both training data (with target) and unseen test data (without
            target).
            """
            self._data = data.copy()

            if drop_nan:
                _before = self._data.shape[0]
                self._data.dropna(inplace=True)
                _after = self._data.shape[0]
                self._display_reduction(_before, _after, "nan")
            self._drop_nan = drop_nan

            if drop_duplicates:
                _before = self._data.shape[0]
                self._data.drop_duplicates(inplace=True)
                _after = self._data.shape[0]
                self._display_reduction(_before, _after, "duplicate")
            self._drop_duplicates = drop_duplicates

            self._target_name = target
            self._target = self._data[target].copy() if target else None

            _exclude = []
            if target:
                _exclude.append(target)
            if drop_id and "id" in self._data.columns:
                _exclude.append("id")
            self._drop_id = drop_id

            if _exclude:
                self._data.drop(columns=_exclude, inplace=True, errors="ignore")

            self._X = {"train": None, "valid": None}
            self._y = {"train": None, "valid": None}

            self._transformer = None
            self._label_encoder = None
            self._split_info = {
                "test_size": 0.0,
                "stratify": False,
                "random_state": 42,
            }

        def _display_reduction(self, before: int, after: int, type_str: str) -> None:
            print(
                f"Preprocessor: Dropping {type_str} values: {before} -> {after} rows (-{((before-after)/before)*100:0.1f}%)"
            )

        def get_data(self):
            """Returns transformed features (and target labels if available)."""
            if self._target_name is not None:
                return self._X, self._y
            return self._data

        def get_metadata(self) -> dict:
            if self._transformer is None:
                raise ValueError(
                    "Transformer has not been fitted yet. Call apply_transform() first."
                )
            return {
                "drop_id": self._drop_id,
                "drop_nan": self._drop_nan,
                "drop_duplicates": self._drop_duplicates,
                "split_info": self._split_info,
                "transformer": self._transformer.transformers_,
            }

        def split_data(
            self,
            test_size: float = 0.2,
            stratify: bool = True,
            random_state: int = 42,
        ) -> None:
            if self._target is None:
                print(
                    "Preprocessor: Cannot split dataset without target column.",
                    file=sys.stderr,
                )
                return

            (
                self._X["train"],
                self._X["valid"],
                self._y["train"],
                self._y["valid"],
            ) = train_test_split(
                self._data,
                self._target,
                test_size=test_size,
                stratify=self._target if stratify else None,
                random_state=random_state,
            )

            self._split_info["test_size"] = test_size
            self._split_info["stratify"] = stratify
            self._split_info["random_state"] = random_state

        def encode_target(
            self, inplace: bool = True, as_series: bool = True
        ) -> dict:
            """Encodes target strings into integers using a single fitted LabelEncoder."""
            if self._target is None:
                print(
                    "Preprocessor: No target present to encode.", file=sys.stderr
                )
                return None

            _y = self._y if inplace else self._y.copy()

            self._label_encoder = LabelEncoder()
            # FIT ONLY ON TRAIN, TRANSFORM VALIDATION TO PREVENT LEAKAGE
            _y["train"] = self._label_encoder.fit_transform(_y["train"])
            _y["valid"] = self._label_encoder.transform(_y["valid"])

            if as_series:
                _y["train"] = pd.Series(
                    _y["train"], index=self._X["train"].index, name=self._target_name
                )
                _y["valid"] = pd.Series(
                    _y["valid"], index=self._X["valid"].index, name=self._target_name
                )

            return None if inplace else _y

        def _convert_to_df(
            self, data_array: np.ndarray, transformer: ColumnTransformer
        ) -> pd.DataFrame:
            """Utility using Scikit-Learn's native get_feature_names_out()."""
            try:
                feature_names = transformer.get_feature_names_out()
            except Exception:
                # Fallback if get_feature_names_out fails
                feature_names = [f"feat_{i}" for i in range(data_array.shape[1])]

            return pd.DataFrame(data_array, columns=feature_names)

        def apply_transform(
            self,
            transformer: ColumnTransformer,
            inplace: bool = True,
            fitted_transformer: ColumnTransformer = None,
        ):
            """Fits and transforms training/validation data OR transforms test data using a pre-fitted transformer.

            Parameters
            ----------
            transformer : ColumnTransformer
                Unfitted ColumnTransformer to fit on training data.
            inplace : bool
                Whether to update class internal state.
            fitted_transformer : ColumnTransformer, optional
                Pre-fitted transformer to transform unseen test data.
            """
            if fitted_transformer is not None:
                # TRANSFORM UNSEEN TEST DATA WITH PRE-FITTED TRANSFORMER
                self._transformer = fitted_transformer
                transformed_arr = self._transformer.transform(self._data)
                df_transformed = self._convert_to_df(
                    transformed_arr, self._transformer
                )

                if inplace:
                    self._data = df_transformed
                    return None
                return df_transformed

            if transformer is None:
                print("Preprocessor: transformer is missing.", file=sys.stderr)
                return None

            self._transformer = transformer

            if self._target_name is not None and self._X["train"] is not None:
                _X = self._X if inplace else self._X.copy()

                # FIT ON TRAIN, TRANSFORM VALID
                arr_train = self._transformer.fit_transform(_X["train"])
                arr_valid = self._transformer.transform(_X["valid"])

                _X["train"] = self._convert_to_df(arr_train, self._transformer)
                _X["valid"] = self._convert_to_df(arr_valid, self._transformer)

                return None if inplace else _X
            else:
                # Unsplit data case
                arr_data = self._transformer.fit_transform(self._data)
                df_data = self._convert_to_df(arr_data, self._transformer)

                if inplace:
                    self._data = df_data
                    return None
                return df_data

    return (Preprocessor,)


@app.cell
def _(clone, np, pd, permutation_importance, sys):
    class CrossValidator:
        def __init__(
            self,
            models: list,
            metric_fns: list,
            cv_method,
            name: str = None,
            pi_kwargs: dict = None,
            pred_probs: bool = False,
            verbose: bool = True,
            sample_weight_fn=None,
            use_eval_set: bool = False,
            fit_params: dict = None,
            eval_weight_keys: dict = None,
        ):
            """
            A class for performing cross-validation on a set of models with various metric functions.

            Attributes:
                _models (list): A list of (name, model) tuples.
                _metric_fns (list): A list of (name, metric_fn) tuples.
                _cv_method (object): A cross-validation splitter from scikit-learn.
                _name (str): A label for this cross-validator, used in logging.
                _pi_kwargs (dict, optional): Keyword arguments for permutation importance.
                _pred_probs (bool): Whether to predict probabilities instead of class labels.
                _sample_weight_fn (callable, optional): Given y (train or eval fold), returns
                    per-row sample weights, e.g. functools.partial(compute_sample_weight, "balanced").
                    WARNING: do not also set class_weight="balanced" (or similar) on the model
                    itself — applying both is double-weighting and has been measured to cost
                    0.04–0.06 balanced accuracy versus picking one mechanism.
                _use_eval_set (bool): If True, passes eval_set=[(x_test, y_test)] to each model's
                    fit() call for early stopping. Only used during CV folds — never during
                    refit_predict(), since using the final held-out set as an eval_set would
                    leak it into training decisions. Not all models support this — e.g.
                    LogisticRegression has no eval_set concept and will raise a TypeError if
                    included in a CrossValidator configured with use_eval_set=True.
                _fit_params (dict, optional): Extra static kwargs merged into every fit() call,
                    e.g. {"verbose": False}.
                _eval_weight_keys (dict): Manual override mapping a model's class name to the
                    fit() kwarg its library uses for per-row eval-set sample weights, since
                    this isn't standardized across libraries — e.g. XGBoost uses
                    "sample_weight_eval_set", LightGBM uses "eval_sample_weight". Only needed
                    for libraries not already handled by _eval_weight_key()'s defaults.
                _perm_imp (dict, optional): Permutation importances per model, set after fit().
                _oof_preds (dict, optional): Out-of-fold predictions per model, set after fit().
                _oof_metrics (dict, optional): Out-of-fold metric scores per model, set after fit().
                _data (tuple, optional): (X, y) actually used across folds, set after fit().
                _oof_metrics_df (pd.DataFrame, optional): Mean out-of-fold scores per model.
                _refit_models (list, optional): (name, model) pairs refit on the full
                    training set, set after refit_predict().

            Methods:
                get_data(): Returns (oof_preds, oof_metrics, fold_data) after fitting.
                get_metadata(): Returns configuration metadata for this cross-validator.
                get_oof_metrics_df(): Returns the mean out-of-fold metrics as a DataFrame.
                get_models(): Returns models as fitted at the end of the last CV fold.
                fit(X, y): Runs cross-validation and stores results.
                predict(X): Predicts using each model's last-fold fitted state.
                refit_predict(X_train, y_train, X_valid): Refits each model on the full
                    training set, then predicts on a held-out validation set.
            """
            self._name = name or self.__class__.__name__
            self._verbose = verbose

            self._models = models
            self._metric_fns = metric_fns
            self._cv_method = cv_method
            self._pi_kwargs = pi_kwargs
            self._pred_probs = pred_probs
            self._sample_weight_fn = sample_weight_fn
            self._use_eval_set = use_eval_set
            self._fit_params = fit_params or {}
            self._eval_weight_keys = eval_weight_keys or {}

            self._perm_imp = None
            self._oof_preds = None
            self._oof_metrics = None
            self._data = None
            self._oof_metrics_df = None
            self._refit_models = None

        def _index(self, data, idx: np.ndarray):
            """Index into a DataFrame/Series or ndarray the correct way for each."""
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return data.iloc[idx]
            return data[idx]

        def _eval_weight_key(self, model) -> str:
            """XGBoost and LightGBM name their eval_set sample-weight argument
            differently in their sklearn-compatible fit() methods. Detect by class
            name; fall back to a manual override via eval_weight_keys if given."""
            cls_name = type(model).__name__
            if cls_name in self._eval_weight_keys:
                return self._eval_weight_keys[cls_name]
            if cls_name.startswith("XGB"):
                return "sample_weight_eval_set"
            return "eval_sample_weight"  # LightGBM's name; also a reasonable default elsewhere

        def _build_fit_kwargs(self, model, x_test=None, y_train=None, y_test=None) -> dict:
            kwargs = dict(self._fit_params)

            if self._sample_weight_fn is not None:
                kwargs["sample_weight"] = self._sample_weight_fn(y_train)

            if self._use_eval_set and x_test is not None and y_test is not None:
                kwargs["eval_set"] = [(x_test, y_test)]
                if self._sample_weight_fn is not None:
                    kwargs[self._eval_weight_key(model)] = [self._sample_weight_fn(y_test)]

            return kwargs

        def _calculate_metrics(self, y_test, y_pred) -> dict:
            # Dictionary to store the score for each metric
            results = {}

            # Loop through each metric
            for metric_name, metric_fn in self._metric_fns:
                try:
                    score = metric_fn(y_test, y_pred)
                except ValueError as e:
                    print(f"{self._name}: failed to compute '{metric_name}' — {e}\n", file=sys.stderr)
                    continue

                # Store score as value and metric as key
                results[metric_name] = score

                if self._verbose:
                    # Display metric score
                    print(f' - {metric_name} : {score:.5f}\n')

            return results

        def _cross_validate(self, X, y) -> tuple:
            # Dictionaries to store out-of-fold predictions and out-of-fold metric scores
            oof_preds, oof_metrics = {}, {}

            # Lists to aggregate test features and labels used in each fold
            x_data, y_data = [], []

            # Dictionary to store permutation feature importance for each fold
            perm_imp = {}

            if self._verbose:
                print(f'Name: {self._name} | {self._cv_method.n_splits}-Fold\n')

            for idx, (train_idx, test_idx) in enumerate(self._cv_method.split(X, y)):
                if self._verbose:
                    print(f'Fold {idx}:')
                    print('-'*40+'\n')

                x_train, x_test = self._index(X, train_idx), self._index(X, test_idx)
                y_train, y_test = self._index(y, train_idx), self._index(y, test_idx)

                x_data.extend(x_test.to_numpy() if isinstance(x_test, pd.DataFrame) else x_test)
                y_data.extend(y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test)

                for model_name, model in self._models:
                    if self._verbose:
                        print(f'Cross-validating: [{model_name}]\n')

                    fit_kwargs = self._build_fit_kwargs(model, x_test=x_test, y_train=y_train, y_test=y_test)
                    model.fit(x_train, y_train, **fit_kwargs)

                    if model_name not in oof_preds:
                        oof_preds[model_name] = []
                        oof_metrics[model_name] = {}
                        perm_imp[model_name] = []

                    y_pred = model.predict_proba(x_test) if self._pred_probs else model.predict(x_test)

                    # Save model predictions
                    oof_preds[model_name].append(y_pred)

                    # Calculate metrics
                    for metric_name, score in self._calculate_metrics(y_test, y_pred).items():
                        oof_metrics[model_name].setdefault(metric_name, []).append(score)

                    # Calculate permutation importances
                    if self._pi_kwargs is not None:
                        if self._verbose:
                            print(' -- Calculating Permutation Importances...\n')
                        try:
                            perm_result = permutation_importance(model, x_test, y_test, **self._pi_kwargs)
                            perm_imp[model_name].append(perm_result.importances)
                        except Exception as e:
                            print(f"{self._name}: permutation importance failed for '{model_name}' — {e}\n", file=sys.stderr)

            if self._pi_kwargs is not None:
                self._perm_imp = perm_imp

            return oof_preds, oof_metrics, (x_data, y_data)

        def _build_oof_metrics_df(self, metric_dict: dict) -> pd.DataFrame:
            records = {'models': []}

            for model_name, metrics in metric_dict.items():
                records['models'].append(model_name)
                for metric_name, scores in metrics.items():
                    records.setdefault(metric_name, []).append(np.mean(scores))

            return pd.DataFrame(records).set_index('models')

        def get_data(self) -> tuple:
            return (self._oof_preds, self._oof_metrics, self._data)

        def get_metadata(self) -> dict:
            return {
                "name": self._name,
                "n_splits": getattr(self._cv_method, "n_splits", None),
                "pred_probs": self._pred_probs,
                "pi_kwargs": self._pi_kwargs,
                "sample_weighted": self._sample_weight_fn is not None,
                "use_eval_set": self._use_eval_set,
                "fit_params": self._fit_params,
                "eval_weight_keys": self._eval_weight_keys,
                "models": [name for name, _ in self._models],
                "metrics": [name for name, _ in self._metric_fns],
            }

        def get_oof_metrics_df(self) -> pd.DataFrame:
            return self._oof_metrics_df

        def get_metric_fns(self) -> list:
            return self._metric_fns

        def get_models(self) -> list:
            """Return the (name, model) pairs as fitted at the end of the last CV fold.
            Note: these reflect only the final fold's training data, not the full
            training set — use refit_predict() if you need models trained on everything.
            """
            return self._models

        def fit(self, X, y) -> None:
            self._oof_preds, self._oof_metrics, self._data = self._cross_validate(X, y)
            self._oof_metrics_df = self._build_oof_metrics_df(self._oof_metrics)

        def predict(self, X) -> dict:
            """Predict using each model's current fitted state (last fold's fit).
            Convenience only — see the fold-state caveat in get_models()."""
            if self._oof_preds is None:
                print(f"{self._name}: call fit() before predict().", file=sys.stderr)
                return {}

            preds = {}
            for model_name, model in self._models:
                preds[model_name] = model.predict_proba(X) if self._pred_probs else model.predict(X)
            return preds

        def refit_predict(self, X_train, y_train, X_valid) -> dict:
            """Refit fresh copies of each model on the full training set, then predict
            on a held-out validation set. This is the correct method for a final,
            untouched-until-now validation check.

            Note: intentionally does NOT use an eval_set/early stopping here, even if
            use_eval_set=True was set for CV — X_valid must stay unseen by training
            decisions, or it stops being a trustworthy final check. sample_weight_fn
            (if set) is still applied to X_train.
            """
            preds = {}
            self._refit_models = []

            fit_kwargs = dict(self._fit_params)
            if self._sample_weight_fn is not None:
                fit_kwargs["sample_weight"] = self._sample_weight_fn(y_train)

            for model_name, model in self._models:
                fresh_model = clone(model)
                fresh_model.fit(X_train, y_train, **fit_kwargs)
                self._refit_models.append((model_name, fresh_model))

                preds[model_name] = (
                    fresh_model.predict_proba(X_valid) if self._pred_probs else fresh_model.predict(X_valid)
                )

            return preds

    return (CrossValidator,)


@app.cell
def _(CrossValidator, pd):
    def aggregate_predictions(cv:CrossValidator, X:dict, y:dict) -> pd.DataFrame:
        results = {}
        preds = cv.refit_predict(X["train"], y["train"], X["valid"])
        metric_fns = cv.get_metric_fns()
        for model, pred in preds.items():
            results[model] = {}
            for item in metric_fns:
                name, fn = item
                results[model][name] = fn(y["valid"], pred)
        return pd.DataFrame(results)

    return (aggregate_predictions,)


@app.cell
def _(balanced_accuracy_score, f1_score, np, partial, train_data):
    ord_map = {
        'sleep_quality': ["poor", "average", "good"],
        "physical_activity_level": ["sedentary", "moderate", "active"],
        "stress_level": ["low", "medium", "high"]
    }

    ord_categories = list(ord_map.values())

    ord_cols = list(ord_map.keys())
    num_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()[1:]

    # Categorical 
    cat_cols = []
    for _col in train_data.select_dtypes(exclude=[np.number]).columns.tolist():
        if (_col not in ord_cols) and _col != "health_condition":
            cat_cols.append(_col)

    feat_map = {
        "num":num_cols,
        "ord":ord_cols,
        "cat":cat_cols,
    }

    metric_fns = [
        ("Balanced Accuracy", balanced_accuracy_score),
        ("Macro F1 ", partial(f1_score, average="macro")), 
    ]
    return cat_cols, metric_fns, num_cols, ord_categories, ord_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Models
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    Preprocessor,
    RobustScaler,
    SimpleImputer,
    cat_cols,
    num_cols,
    ord_categories,
    ord_cols,
    train_data,
):
    lr_trans = ColumnTransformer(
        transformers=[
            # Numerical Pipeline
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy="median", add_indicator=True)),
                ('scaler', RobustScaler())
            ]), num_cols),

            # Ordinal Pipeline
            ('ord', Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OrdinalEncoder(
                        categories=ord_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]), ord_cols),
        
            # Categorical Pipeline 
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    # Naive approach (Logistic Regression)
    lr_pp = Preprocessor(train_data, "health_condition")
    lr_pp.split_data()
    lr_pp.apply_transform(lr_trans, inplace=True)
    lr_pp.encode_target(inplace=True, as_series=True)

    X_lr, y_lr = lr_pp.get_data()
    return X_lr, y_lr


@app.cell
def _(X_lr):
    X_lr["train"].head()
    return


@app.cell
def _(y_lr):
    y_lr["train"].head()
    return


@app.cell
def _(
    CrossValidator,
    LogisticRegression,
    SGDClassifier,
    StratifiedKFold,
    X_lr,
    metric_fns,
    y_lr,
):
    lr_models = [
        ("Logistic Regression", LogisticRegression(random_state=42)),
        ("SGD Classifier", SGDClassifier(random_state=42))
    ]

    lr_cv = CrossValidator(
        models=lr_models,
        metric_fns=metric_fns,
        cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        name = "Linear Model CV"
    )

    lr_cv.fit(X_lr["train"], y_lr["train"])
    lr_cv.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return (lr_cv,)


@app.cell
def _(X_lr, aggregate_predictions, lr_cv, y_lr):
    aggregate_predictions(lr_cv, X_lr, y_lr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tree-based Models
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    Preprocessor,
    SimpleImputer,
    cat_cols,
    num_cols,
    ord_categories,
    ord_cols,
    train_data,
):
    tree_trans = ColumnTransformer(
        transformers=[
            # Numerical Pipeline
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy="median", add_indicator=True)),
            ]), num_cols),
        
            # Ordinal Pipeline
            ('ord', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(
                        categories=ord_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]), ord_cols),
        
            # Categorical Pipeline 
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ],
        remainder="passthrough"
    ) 

    tree_pp = Preprocessor(train_data, "health_condition")
    tree_pp.split_data()
    tree_pp.apply_transform(tree_trans, inplace=True)
    tree_pp.encode_target(inplace=True, as_series=True)

    X_tree, y_tree = tree_pp.get_data()
    return X_tree, y_tree


@app.cell
def _(X_tree):
    X_tree["train"].head()
    return


@app.cell
def _(y_tree):
    y_tree["valid"].head()
    return


@app.cell
def _(
    CrossValidator,
    DecisionTreeClassifier,
    StratifiedKFold,
    X_tree,
    metric_fns,
    y_tree,
):
    tree_models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ]

    tree_cv = CrossValidator(
        models=tree_models,
        metric_fns=metric_fns,
        cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        name = "Tree-based Model CV"
    )

    tree_cv.fit(X_tree["train"], y_tree["train"])
    tree_cv.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return (tree_cv,)


@app.cell
def _(X_tree, aggregate_predictions, tree_cv, y_tree):
    aggregate_predictions(tree_cv, X_tree, y_tree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradient-boosting Models
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    Preprocessor,
    cat_cols,
    ord_categories,
    ord_cols,
    train_data,
):
    gb_trans = ColumnTransformer(
        transformers=[
            # Ordinal Pipeline
            ('ord', Pipeline([
                ('encoder', OrdinalEncoder(
                        categories=ord_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]), ord_cols),
        
            # Categorical Pipeline 
            ('cat', Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ],
        remainder="passthrough"
    ) 
    gb_pp = Preprocessor(train_data, "health_condition")
    gb_pp.split_data()
    gb_pp.apply_transform(gb_trans, inplace=True)
    gb_pp.encode_target(inplace=True, as_series=True)

    X_gb, y_gb = gb_pp.get_data()
    return X_gb, gb_trans, y_gb


@app.cell
def _(X_gb):
    X_gb["train"].head()
    return


@app.cell
def _(y_gb):
    y_gb["train"].head()
    return


@app.cell
def _(
    CrossValidator,
    LGBMClassifier,
    StratifiedKFold,
    XGBClassifier,
    X_gb,
    metric_fns,
    y_gb,
):
    gb_models = [
        ("XGBoost", XGBClassifier(verbosity=0, random_state=42)),
        ("LightGBM", LGBMClassifier(verbose=0, random_state=42)),
        # ("CatBoost", CatBoostClassifier(verbose=False, random_state=42))
    ]

    gb_cv_1 = CrossValidator(
        models=gb_models,
        metric_fns=metric_fns,
        cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        name="Gradient-boost Model CV",
    )

    gb_cv_1.fit(X_gb["train"], y_gb["train"])
    gb_cv_1.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return gb_cv_1, gb_models


@app.cell
def _(X_gb, aggregate_predictions, gb_cv_1, y_gb):
    aggregate_predictions(gb_cv_1, X_gb, y_gb)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class Weights
    """)
    return


@app.cell
def _(
    CrossValidator,
    StratifiedKFold,
    X_gb,
    compute_sample_weight,
    gb_models,
    metric_fns,
    partial,
    y_gb,
):
    gb_cv_2 = CrossValidator(
        models=gb_models,
        metric_fns=metric_fns,
        cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        sample_weight_fn=partial(compute_sample_weight, "balanced"),
        use_eval_set=True,
        name="Class-aware Gradient-Boost Model CV"
    )

    gb_cv_2.fit(X_gb["train"], y_gb["train"])
    gb_cv_2.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return (gb_cv_2,)


@app.cell
def _(X_gb, aggregate_predictions, gb_cv_2, y_gb):
    aggregate_predictions(gb_cv_2, X_gb, y_gb)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make Predictions
    """)
    return


@app.cell
def _(main_dir, pd):
    sample_sub = pd.read_csv(f"{main_dir}/sample_submission.csv")
    sample_sub.head()
    return


@app.cell
def _(Preprocessor, gb_trans, test_data):
    test_pp = Preprocessor(test_data)
    test_pp.apply_transform(gb_trans, inplace=True)
    test_pp.get_data()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
