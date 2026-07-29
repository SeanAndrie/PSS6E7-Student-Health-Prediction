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
    # Import Dependencies
    """)
    return


@app.cell
def _():
    import kagglehub
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, LabelEncoder
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    from xgboost import XGBClassifier

    return (
        ColumnTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RobustScaler,
        SimpleImputer,
        pd,
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
    main_dir = "playground-series-s6e7/"

    train_data = pd.read_csv(f"{main_dir}/train.csv")
    test_data = pd.read_csv(f"{main_dir}/test.csv")
    sample_sub = pd.read_csv(f"{main_dir}/sample_submission.csv")

    train_data.shape, test_data.shape
    return (train_data,)


@app.cell
def _(ColumnTransformer, pd, train_test_split):
    class Preprocessor:
        def __init__(
            self,
            data:pd.DataFrame,
            target:str,
            split_percent:float = 0.2,
            drop_id:bool = True,
            test_size:float = 0.2,
            stratify:bool = True,
            random_state:int = 42,
        ):
            self._data = data
            self._target = target
            self._drop_id = drop_id
            self._stratify = stratify
            self._test_size = test_size
            self._random_state = random_state
            self._split_percent = split_percent
        
            self._X:dict = {"train": None, "valid": None}
            self._y:dict = {"train": None, "valid": None}
            self._transformer:ColumnTransformer = None

            self._split_data()

        def get_data(self) -> tuple[dict, dict]:
            return (self._X, self._y)

        def get_transformer(self) -> ColumnTransformer:
            return self._transformer
        
        def transform_data(self, transformer:ColumnTransformer) -> None:
            self._X["train"] = transformer.fit_transform(self._X["train"])
            self._X["valid"] = transformer.transform(self._X["valid"])
            self._transformer = transformer
    
        def _split_data(self) -> None:
            drop_cols = [self._target]
            if self._drop_id:
                drop_cols += ["id"]
            X = self._data.drop(drop_cols, axis=1)
            y = self._data[self._target]
        
            self._X["train"], self._X["valid"], self._y["train"], self._y["valid"] = train_test_split(X, y, test_size=self._test_size, random_state=self._random_state, stratify=y if self._stratify else None)        

    return (Preprocessor,)


@app.cell
def _(Preprocessor, train_data):
    preproc = Preprocessor(train_data, "health_condition")
    X, y = preproc.get_data()
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    RobustScaler,
    SimpleImputer,
    categorical_cols,
    numerical_cols,
    ordinal_categories,
    ordinal_cols,
    preprocessor,
):
    transformer = ColumnTransformer(
        transformers=[
            # Numerical pipeline: median imputation → missing indicator → robust scaling
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('scaler', RobustScaler())
            ]), numerical_cols),

            # Ordinal pipeline: most-frequent imputation + OrdinalEncoder
            ('ord', Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OrdinalEncoder(
                        categories=ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                )
            ]), ordinal_cols),

            # Categorical pipeline: most-frequent imputation → one-hot encoding
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ]
    )

    print(preprocessor)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
