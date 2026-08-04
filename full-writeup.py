import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PS-S6E7 | Predicting Student Health Risk: A Simple Solution Walkthrough
    This write-up covers a simple approach to the Kaggle Playground Series competition "Predicting Student Health Risk" (Season 6, Episode 7).

    ## About the dataset
    This dataset is a comprehensive student health dataset containing 50,000 records, created to capture the wide-ranging lifestyle, physiological, and psychological characteristics of college students. It is modeled on trends observed in large-scale student health studies conducted in China, ensuring realistic variation in daily habits and health-related attributes. The dataset integrates information collected from surveys, wearable devices, and institutional health records, encompassing lifestyle behaviors, psychological indicators, and time-series data.

    Each record corresponds to a time-stamped observation, capturing how student habits and conditions change over time. The dataset integrates multiple dimensions of health, including physical activity, sleep behavior, mental well-being, Chronic Fatigue, and academic influences, providing a comprehensive view of factors affecting student health.

    ## Feature information

    1. **Categorical**
        - `health_condition` - Target variable. Overall health classification of the student: Fit/At-Risk/Unhealthy
        - `diet_type` - The student's general eating pattern/category (e.g., balanced, high-carb, protein, junk food, vegetarian, etc.).
        - `smoking_alcohol` - Indicates whether the student smokes and/or consumes alcohol (e.g., None, Smoking only, Alcohol only, Both).
        - `gender` - The student's gender (e.g., Male, Female, Other).

    2. **Ordinal**
        - `sleep_quality` - A ranked rating of how restful/effective the student's sleep is (e.g., Poor -> Fair -> Good -> Excellent).
        - `physical_activity_level` - A ranked category of how physically active the student is overall (e.g., Sedentary -> Light -> Moderate-> Active -> Very Active).
        - `stress_level` - A ranked measure of the student's perceived stress (e.g., Low -> Moderate -> High -> Severe).

    3. **Numerical**
        - `sleep_duration` - Total hours of sleep per night.
        - `heart_rate` - Resting or average heart rate in beats per minute (bpm).
        - `bmi` - Body Mass Index.
        - `calorie_expenditure` - Estimated total calories burned per day.
        - `step_count` - Number of steps take per day.
        - `exercise_duration` - Total minutes spent in dedicated exercise/workout sessions per day.
        - `water_intake` - Daily water consumption.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import Dependencies
    """)
    return


@app.cell
def _():
    import os
    import sys
    import kagglehub
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from IPython import display
    from functools import partial

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, LabelEncoder
    from sklearn.inspection import permutation_importance

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    return (
        ColumnTransformer,
        LabelEncoder,
        clone,
        kagglehub,
        np,
        pd,
        permutation_importance,
        plt,
        sns,
        sys,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(
    ColumnTransformer,
    LabelEncoder,
    clone,
    np,
    pd,
    permutation_importance,
    sys,
    train_test_split,
):
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
        ):
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
            self, model_name, model, x_test=None, y_train=None, y_test=None
        ) -> dict:
            kwargs = dict(self._fit_params)

            if self._sample_weight_fn is not None:
                kwargs["sample_weight"] = self._sample_weight_fn(y_train)

            if self._use_eval_set and x_test is not None and y_test is not None:
                kwargs["eval_set"] = [(x_test, y_test)]
                if model_name == "XGBoost":
                    kwargs["verbose"] = False
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
                        model_name, model, x_test=x_test, y_train=y_train, y_test=y_test
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Data
    """)
    return


@app.cell
def _(kagglehub, pd, sns):
    """
    If you have a Kaggle API token, uncomment the line below to download the dataset using the KaggleHub API
    """
    # kagglehub.login()

    try: 
        path = kagglehub.competition_download('playground-series-s6e7')
    except Exception as e: 
        print("Failed to download competition data. Manually download the dataset from the competitions page.")
    else:
        print(f"-- Downloaded competition data to '{path}' --\n")

    main_dir = "playground-series-s6e7"

    try:
        train_data = pd.read_csv(f"{main_dir}/train.csv")
        test_data = pd.read_csv(f"{main_dir}/test.csv")
        sample_sub = pd.read_csv(f"{main_dir}/sample_submission.csv")
    except FileNotFoundError:
        print(f"Failed to load data. Ensure that '{main_dir}' exists in your current working directory.")
    else: 
        print(f"-- '{main_dir}' found. Loaded '{main_dir}' successfully --")

    sns.set_theme(style="darkgrid", palette="Set2")
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis

    ## First glances at the data
    """)
    return


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell
def _(test_data):
    test_data.head()
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell
def _(test_data):
    test_data.describe()
    return


@app.cell
def _(train_data):
    train_data.info()
    return


@app.cell
def _(test_data):
    test_data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target Analysis

    Start with the target. `health_condition` classifies each student into one of three buckets: `fit`, `at-risk`, or `unhealthy`. Let's look at how these three classes are distributed in the training data.

    ### Target Distribution
    """)
    return


@app.cell
def _(plt, sns, train_data):
    classes = train_data["health_condition"].unique().tolist()
    class_cnts = {cls : len(train_data[train_data["health_condition"] == cls]) for cls in classes}

    fig_1, ax_1 = plt.subplots(1, 2, figsize=(15, 8))

    ax_1[0].pie(x=list(class_cnts.values()), labels=list(class_cnts.keys()), autopct="%1.1f%%")
    sns.countplot(data=train_data, x="health_condition", hue="health_condition", ax=ax_1[1])

    fig_1.tight_layout()
    plt.gca()
    return (classes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: That's a severe class imbalance. Nearly nine out of ten students fall into `at-risk`. This single fact should shape almost everything you do next, and it's why the competition's metric matters so much.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics: Plain accuracy vs. Balanced Accuracy

    The competition score submissions with **Balanced Accuracy**. Balanced accuracy averages the *recall* of each class instead of counting overall correct predictions. For $K$ classes, balanced accuracy is equivalent to the *Macro-Averaged Recall* across all classes:

    $$
    \text{Balanced Accuracy} = \frac{1}{K}\sum_{k=1}^{K}{\text{Recall}_k}
    $$

    *Recall* measures the ability of a model to find all actual positive instances, which is calculated as:

    $$
    \text{Recall} = \frac{TP}{TP + FN}
    $$

    where,

    $$
    TP\rightarrow\text{True Positives}
    $$

    $$
    FN\rightarrow\text{False Negatives}
    $$

    Imagine a model that just predicts `at-risk` for every single student, no matter what. Plain accuracy would reward that model with a score of 85.9%, since it happens to guess right most of the time. But its recall on `fit` is 0%, and its recall on `unhealthy` is 0%. Balanced accuracy averages those three recall values: (0 + 100 + 0)/3, which comes out to roughly 33.3%. That's barely better than random guessing across three classes. This tells you the metric will punish any model that leans on the majority class as a shortcut.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1**: Reimplement *Balanced Accuracy* based on the description given above.
    """)
    return


@app.cell
def _(np, pd, train_data):
    def accuracy(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)

        if total_predictions == 0:
            return 0.0
        return correct_predictions / total_predictions

    def class_recall(y_true, y_pred, k):

        TP = np.sum((y_true == k) & (y_pred == k))
        FN = np.sum((y_true == k) & (y_pred != k))

        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def flip_minority(n):
        if n == 0:
            return n
        return n - 1 if n == 2 else n + 1
    
    def balanced_accuracy(y_true, y_pred) -> np.float32:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Add your code here ... 

        return 

    y_true = pd.Series(train_data["health_condition"].astype("category").cat.codes, name="health_condition")

    y_pred = y_true.map(flip_minority)

    print(f"Accuracy: {accuracy(y_true, y_pred) * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy(y_true, y_pred) * 100:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How then do we ensure that our model treats each class the same during training despite the imbalance?
    """)
    return


@app.cell
def _(classes, np, train_data):
    def w_k(samples, k):
        samples = np.asarray(samples)
        N = len(samples)
        K = len(np.unique(samples))
        n_k = len(samples[samples == k])

        return N / (K * n_k)

    weights = {}
    for k in classes:
        weights[k] = (w_k(train_data["health_condition"], k))

    print(weights)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
