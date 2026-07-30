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

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score 

    return (
        ColumnTransformer,
        LGBMClassifier,
        LabelEncoder,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RobustScaler,
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
        plt,
        roc_auc_score,
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
    CONFIG = {
        "FE" : False 
    }

    main_dir = "playground-series-s6e7"

    train_data = pd.read_csv(f"{main_dir}/train.csv")
    test_data = pd.read_csv(f"{main_dir}/test.csv")

    train_data.shape, test_data.shape
    return CONFIG, main_dir, test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Preprocess Data

    ## Preprocessor
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
            return {
                "drop_id": self._drop_id,
                "drop_nan": self._drop_nan,
                "drop_duplicates": self._drop_duplicates,
                "split_info": self._split_info,
                "transformer": self._transformer.transformers_,
                "label_encoder": self._label_encoder
            }

        def split_data(
            self,
            test_size: float = 0.2,
            stratify: bool = True,
            random_state: int = 42,
        ) -> None:
            if self._target is None:
                print("Preprocessor: Cannot split dataset without target column.", file=sys.stderr)
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

        def get_classes(self):
            if self._target is None:
                print("Preprocessor: No target specified.", file=sys.stderr)
                return None

            classes = None
            if self._label_encoder is None:
                print(
                    "Preprocessor: target has not been encoded yet. Call encode target using `encode_target` method to return encoded labels from fitted label encoder.",
                    file=sys.stderr
                )
                classes = self._target.unique().to_numpy()
            else:
                classes = self._label_encoder.classes_

            return classes

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
                feature_names = [f"feat_{i}" for i in range(data_array.shape[1])]

            return pd.DataFrame(data_array, columns=feature_names)

        def apply_transform(
            self,
            transformer: ColumnTransformer,
            inplace: bool = True,
            # fitted_transformer: ColumnTransformer = None,
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
            # if fitted_transformer is not None:
            #     self._transformer = fitted_transformer
            #     transformed_arr = self._transformer.transform(self._data)
            #     df_transformed = self._convert_to_df(
            #         transformed_arr, self._transformer
            #     )

            #     if inplace:
            #         self._data = df_transformed
            #         return None
            #     return df_transformed

            if transformer is None:
                print("Preprocessor: transformer is missing.", file=sys.stderr)
                return None

            self._transformer = transformer

            if self._target_name is not None and self._X["train"] is not None:
                _X = self._X if inplace else self._X.copy()

                arr_train = self._transformer.fit_transform(_X["train"])
                arr_valid = self._transformer.transform(_X["valid"])

                _X["train"] = self._convert_to_df(arr_train, self._transformer)
                _X["valid"] = self._convert_to_df(arr_valid, self._transformer)

                return None if inplace else _X
            else:
                arr_data = self._transformer.transform(self._data)
                df_data = self._convert_to_df(arr_data, self._transformer)

                if inplace:
                    self._data = df_data
                    return None
                return df_data

    return (Preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross-validation
    """)
    return


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
            adjust_priors: bool = False,
            verbose: bool = True,
            sample_weight_fn=None,
            use_eval_set: bool = False,
            fit_params: dict = None,
            eval_weight_keys: dict = None,
        ):
            self._name = name or self.__class__.__name__
            self._verbose = verbose

            self._models = models
            self._metric_fns = metric_fns
            self._cv_method = cv_method
            self._pi_kwargs = pi_kwargs
            self._pred_probs = pred_probs
            self._adjust_priors = adjust_priors
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

        def _compute_priors(self, model, y_train) -> np.ndarray:
            counts = pd.Series(y_train).value_counts(normalize=True)
            return np.array([counts.get(cls, 1e-12) for cls in model.classes_])

        def _predict_model(self, model, X, y_train=None):
            """Generates predictions.

            If adjust_priors=True, evaluates probabilities P(y=k|x), divides by
            priors pi_k, and returns argmax_k(P(y=k|x) / pi_k).
            """
            if self._adjust_priors or self._pred_probs:
                probs = model.predict_proba(X)

                if self._adjust_priors:
                    if y_train is None:
                        raise ValueError(
                            "y_train is required to compute priors for adjust_priors=True."
                        )
                    priors = self._compute_priors(model, y_train)
                    adjusted_probs = probs / (priors + 1e-12)

                    if self._pred_probs:
                        return adjusted_probs / np.sum(
                            adjusted_probs, axis=1, keepdims=True
                        )
                    return np.argmax(adjusted_probs, axis=1)

                return probs

            # Standard argmax prediction
            return model.predict(X)

        def _index(self, data, idx: np.ndarray):
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return data.iloc[idx]
            return data[idx]

        def _eval_weight_key(self, model) -> str:
            cls_name = type(model).__name__
            if cls_name in self._eval_weight_keys:
                return self._eval_weight_keys[cls_name]
            if cls_name.startswith("XGB"):
                return "sample_weight_eval_set"
            return "eval_sample_weight"

        def _build_fit_kwargs(
            self, model, x_test=None, y_train=None, y_test=None
        ) -> dict:
            kwargs = dict(self._fit_params)

            if self._sample_weight_fn is not None:
                kwargs["sample_weight"] = self._sample_weight_fn(y_train)

            if self._use_eval_set and x_test is not None and y_test is not None:
                kwargs["eval_set"] = [(x_test, y_test)]
                if self._sample_weight_fn is not None:
                    kwargs[self._eval_weight_key(model)] = [
                        self._sample_weight_fn(y_test)
                    ]

            return kwargs

        def _calculate_metrics(self, y_test, y_pred) -> dict:
            results = {}
            for metric_name, metric_fn in self._metric_fns:
                try:
                    score = metric_fn(y_test, y_pred)
                except ValueError as e:
                    print(
                        f"{self._name}: failed to compute '{metric_name}' — {e}\n",
                        file=sys.stderr,
                    )
                    continue

                results[metric_name] = score
                if self._verbose:
                    print(f" - {metric_name} : {score:.5f}\n")
            return results

        def _cross_validate(self, X, y) -> tuple:
            oof_preds, oof_metrics = {}, {}
            x_data, y_data = [], []
            perm_imp = {}

            if self._verbose:
                print(f"Name: {self._name} | {self._cv_method.n_splits}-Fold\n")

            for idx, (train_idx, test_idx) in enumerate(
                self._cv_method.split(X, y)
            ):
                if self._verbose:
                    print(f"Fold {idx}:")
                    print("-" * 40 + "\n")

                x_train, x_test = self._index(X, train_idx), self._index(X, test_idx)
                y_train, y_test = self._index(y, train_idx), self._index(y, test_idx)

                x_data.extend(
                    x_test.to_numpy()
                    if isinstance(x_test, pd.DataFrame)
                    else x_test
                )
                y_data.extend(
                    y_test.to_numpy()
                    if isinstance(y_test, pd.Series)
                    else y_test
                )

                for model_name, model in self._models:
                    if self._verbose:
                        print(f"Cross-validating: [{model_name}]\n")

                    fit_kwargs = self._build_fit_kwargs(
                        model, x_test=x_test, y_train=y_train, y_test=y_test
                    )
                    model.fit(x_train, y_train, **fit_kwargs)

                    if model_name not in oof_preds:
                        oof_preds[model_name] = []
                        oof_metrics[model_name] = {}
                        perm_imp[model_name] = []

                    # Predict using fold-specific y_train for prior calculations
                    y_pred = self._predict_model(model, x_test, y_train=y_train)

                    oof_preds[model_name].append(y_pred)

                    for metric_name, score in self._calculate_metrics(
                        y_test, y_pred
                    ).items():
                        oof_metrics[model_name].setdefault(
                            metric_name, []
                        ).append(score)

                    if self._pi_kwargs is not None:
                        if self._verbose:
                            print(" -- Calculating Permutation Importances...\n")
                        try:
                            perm_result = permutation_importance(
                                model, x_test, y_test, **self._pi_kwargs
                            )
                            perm_imp[model_name].append(perm_result.importances)
                        except Exception as e:
                            print(
                                f"{self._name}: permutation importance failed for '{model_name}' — {e}\n",
                                file=sys.stderr,
                            )

            if self._pi_kwargs is not None:
                self._perm_imp = perm_imp

            return oof_preds, oof_metrics, (x_data, y_data)

        def _build_oof_metrics_df(self, metric_dict: dict) -> pd.DataFrame:
            records = {"models": []}
            for model_name, metrics in metric_dict.items():
                records["models"].append(model_name)
                for metric_name, scores in metrics.items():
                    records.setdefault(metric_name, []).append(np.mean(scores))
            return pd.DataFrame(records).set_index("models")

        def fit(self, X, y) -> None:
            self._oof_preds, self._oof_metrics, self._data = self._cross_validate(
                X, y
            )
            self._oof_metrics_df = self._build_oof_metrics_df(self._oof_metrics)

        def refit_predict(self, X_train, y_train, X_valid) -> dict:
            preds = {}
            self._refit_models = []

            fit_kwargs = dict(self._fit_params)
            if self._sample_weight_fn is not None:
                fit_kwargs["sample_weight"] = self._sample_weight_fn(y_train)

            for model_name, model in self._models:
                fresh_model = clone(model)
                fresh_model.fit(X_train, y_train, **fit_kwargs)
                self._refit_models.append((model_name, fresh_model))

                preds[model_name] = self._predict_model(
                    fresh_model, X_valid, y_train=y_train
                )

            return preds

        def get_data(self) -> tuple:
            return (self._oof_preds, self._oof_metrics, self._data)

        def get_metadata(self) -> dict:
            return {
                "name": self._name,
                "n_splits": getattr(self._cv_method, "n_splits", None),
                "pred_probs": self._pred_probs,
                "adjust_priors": self._adjust_priors,
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

        def get_perm_importances(self) -> dict:
            return self._perm_imp

        def get_models(self) -> list:
            return self._models

    return (CrossValidator,)


@app.cell
def _(CrossValidator, np, pd, plt):
    def top_k_permutation_scores(perm_dict, features, k = 5, ncols = 2):
        model_pi_df = pd.DataFrame(index = features)
        for model, importances in perm_dict.items():
            model_pi_df[f'{model}'] = np.mean(np.concatenate(importances, axis = 1), axis = 1)    

        nrows = -(-len(model_pi_df.columns) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize = (18, 8*nrows))
        for idx, model in enumerate(model_pi_df.columns.tolist()):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            model_pi_df[model].sort_values()[-k:].plot(kind = 'barh', ax = ax, title = f'{model} | Top {k} PI Scores')

        if len(perm_dict.keys()) % ncols != 0:
            for j in range(len(perm_dict.keys()) % ncols, ncols):
                axes[-1, j].axis('off')

        fig.tight_layout()

        return model_pi_df

    def aggregate_predictions(cv:CrossValidator, X:dict, y:dict):
        results = {}
        preds = cv.refit_predict(X["train"], y["train"], X["valid"])
        metric_fns = cv.get_metric_fns()
        for model, pred in preds.items():
            results [model] = {}
            for item in metric_fns:
                name, fn = item
                results[model][name] = fn(y["valid"], pred)
        return pd.DataFrame(results), preds

    return aggregate_predictions, top_k_permutation_scores


@app.cell
def _(pd, test_data, train_data):
    def create_interaction_features(df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # df["stress_sleep"] = df["stress_level"].astype(str) + "_" + df["sleep_quality"].astype(str)
        # df["diet_activity"] = df["diet_type"].astype(str) + "_" + df["physical_activity_level"].astype(str)
        # df["stress_activity"] = df["stress_level"].astype(str) + "_" + df["physical_activity_level"].astype(str)
        df["sleep_below_6"] = (df["sleep_duration"]  < 5.995).astype(int)
        df["sleep_below_7"] = (df["sleep_duration"]  < 6.995).astype(int)
        return df

    train_eng = create_interaction_features(train_data)
    test_eng = create_interaction_features(test_data)
    return test_eng, train_eng


@app.cell
def _(train_eng):
    train_eng.head()
    return


@app.cell
def _(test_eng):
    test_eng.head()
    return


@app.cell
def _(
    CONFIG,
    balanced_accuracy_score,
    f1_score,
    np,
    partial,
    test_eng,
    train_data,
    train_eng,
):
    train = train_eng if CONFIG["FE"] else train_data
    test = test_eng if CONFIG["FE"] else test_eng 

    ord_map = {
        'sleep_quality': ["poor", "average", "good"],
        "physical_activity_level": ["sedentary", "moderate", "active"],
        "stress_level": ["high", "medium", "low"]
    }

    ord_categories = list(ord_map.values())

    ord_cols = list(ord_map.keys())
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()[1:]

    # Categorical 
    cat_cols = []
    for _col in train.select_dtypes(exclude=[np.number]).columns.tolist():
        if (_col not in ord_cols) and _col != "health_condition":
            cat_cols.append(_col)
        
    metric_fns = [
        ("Balanced Accuracy", balanced_accuracy_score),
        ("Macro F1 ", partial(f1_score, average="macro")), 
    ]
    return cat_cols, metric_fns, num_cols, ord_categories, ord_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Models
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
    return X_lr, lr_pp, y_lr


@app.cell
def _(X_lr):
    X_lr["train"].head()
    return


@app.cell
def _(y_lr):
    y_lr["train"].head()
    return


@app.cell
def _(lr_pp):
    lr_pp.get_classes()
    return


@app.cell
def _():
    # lr_models = [
    #     ("Logistic Regression", LogisticRegression(random_state=42)),
    #     ("SGD Classifier", SGDClassifier(random_state=42))
    # ]

    # lr_cv = CrossValidator(
    #     models=lr_models,
    #     metric_fns=metric_fns,
    #     cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    #     name = "Linear Model CV"
    # )

    # lr_cv.fit(X_lr["train"], y_lr["train"])
    # lr_cv.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return


@app.cell
def _():
    # aggregate_predictions(lr_cv, X_lr, y_lr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tree-based Models
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
def _():
    # tree_models = [
    #     ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    # ]

    # tree_cv = CrossValidator(
    #     models=tree_models,
    #     metric_fns=metric_fns,
    #     cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    #     name = "Tree-based Model CV"
    # )

    # tree_cv.fit(X_tree["train"], y_tree["train"])
    # tree_cv.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return


@app.cell
def _():
    # aggregate_predictions(tree_cv, X_tree, y_tree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient-boosting Models
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
    return X_gb, gb_pp, gb_trans, y_gb


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
    compute_sample_weight,
    metric_fns,
    partial,
    y_gb,
):
    gb_models = [
        ("XGBoost", XGBClassifier(verbosity=0, device="cuda", random_state=42)),
        ("LightGBM", LGBMClassifier(verbose=0, device="gpu", random_state=42)),
        # ("CatBoost", CatBoostClassifier(verbose=False, task_type="GPU", random_state=42))
    ]

    gb_base = CrossValidator(
        models=gb_models,
        metric_fns=metric_fns,
        sample_weight_fn=partial(compute_sample_weight, "balanced"),
        cv_method=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        pi_kwargs = {'scoring':'balanced_accuracy', 'random_state': 42, 'n_jobs': 4, 'n_repeats': 3},
        name="Class-weighted CV",
    )

    gb_base.fit(X_gb["train"], y_gb["train"])
    gb_base.get_oof_metrics_df().sort_values(by="Balanced Accuracy", ascending=False)
    return gb_base, gb_models


@app.cell
def _(X_gb, gb_base, plt, top_k_permutation_scores):
    gb_base_pi = top_k_permutation_scores(gb_base.get_perm_importances(), X_gb["train"].columns.tolist(), k=10, ncols=2)
    gb_base_pi.head()
    plt.gca()
    return


@app.cell
def _(X_gb, aggregate_predictions, gb_base, y_gb):
    base_results, base_preds = aggregate_predictions(gb_base, X_gb, y_gb)
    base_results
    return (base_results,)


@app.cell
def _(
    CrossValidator,
    StratifiedKFold,
    X_gb,
    aggregate_predictions,
    compute_sample_weight,
    gb_models,
    partial,
    roc_auc_score,
    y_gb,
):
    gb_blend = CrossValidator(
        models=gb_models,
        metric_fns=[("ROC-AUC", partial(roc_auc_score, average="macro", multi_class="ovr"))],
        sample_weight_fn=partial(compute_sample_weight, "balanced"),
        cv_method=StratifiedKFold(n_splits=7, shuffle=True, random_state=42),
        pred_probs=True,
        name="Class-weighted CV",
    )

    blend_results, blend_preds = aggregate_predictions(gb_blend, X_gb, y_gb)
    blend_results
    return blend_preds, blend_results, gb_blend


@app.cell
def _(
    balanced_accuracy_score,
    base_results,
    blend_preds,
    blend_results,
    np,
    y_gb,
):
    def blend_predictions(model_probs:dict, weights:np.array) -> np.array:
        probas = np.array([value for value in model_probs.values()])
        blended_proba = np.average(probas, axis=0, weights=weights)
        return np.argmax(blended_proba, axis=1)

    weights = np.array([blend_results[model]["ROC-AUC"] for model in blend_results]) ** 20
    weights /= weights.sum()

    blended_pred = blend_predictions(blend_preds, weights)
    base_results["Blended"] = balanced_accuracy_score(y_gb["valid"], blended_pred)
    base_results
    return blend_predictions, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submission
    """)
    return


@app.cell
def _(main_dir, pd):
    sample_sub = pd.read_csv(f"{main_dir}/sample_submission.csv")
    sample_sub.head()
    return (sample_sub,)


@app.cell
def _(Preprocessor, gb_trans, test_data):
    test_pp = Preprocessor(test_data)
    test_pp.apply_transform(gb_trans, inplace=True)
    test_data_preproc = test_pp.get_data()
    test_data_preproc.head()
    return (test_data_preproc,)


@app.cell
def _(X_gb, blend_predictions, gb_blend, test_data_preproc, weights, y_gb):
    preds = gb_blend.refit_predict(X_gb["train"], y_gb["train"], test_data_preproc)
    final_preds = blend_predictions(preds, weights)
    final_preds
    return (final_preds,)


@app.cell
def _(final_preds, gb_pp, sample_sub):
    submission = sample_sub.copy()
    submission["health_condition"] = final_preds
    submission["health_condition"] = submission["health_condition"].map(lambda idx: gb_pp.get_classes()[idx])

    id_col = sample_sub["id"]
    submission.drop("id", axis=1, inplace=True)
    submission.index = id_col

    submission.head()
    submission.to_csv("submission.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
