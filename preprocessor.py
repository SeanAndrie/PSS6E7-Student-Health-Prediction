import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import kagglehub
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    return (
        ColumnTransformer,
        LabelEncoder,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RobustScaler,
        SimpleImputer,
        np,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Download the dataset only if you don't have it
    """)
    return


@app.cell
def _():
    main_dir = "playground-series-s6e7"
    return (main_dir,)


@app.cell
def _(main_dir, pd):
    try:
        train_data = pd.read_csv(main_dir + "/train.csv")
        test_data = pd.read_csv(main_dir + "/test.csv")
        sample_submission = pd.read_csv(main_dir + "/sample_submission.csv")
    except FileNotFoundError:
        print("Failed to read data")
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic data display for viewing
    """)
    return


@app.cell
def _(train_data):
    train_data.info()
    return


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell
def _(train_data):
    train_data['health_condition'].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop the 'id' column which lacks a predictive property as it may influence model's prediction
    """)
    return


@app.cell
def _(train_data):
    train_data_1 = train_data.drop(columns='id')
    return (train_data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drop the target column so that it doesn't leak the result into the prediction.

    - X is the train_data without the health_condition column
    - y is only the health_condition column of the train_data
    - X_test is the test_data which already doesn't contain the health_condition column
    """)
    return


@app.cell
def _(test_data, train_data_1):
    X = train_data_1.drop(columns=['health_condition'])  # drop the target column
    y = train_data_1['health_condition']  # keep the target column
    X_test = test_data.copy()  # test data doesn't contain health_condition
    return X, X_test, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    80/20 split for testing and verification data
    - X_train is the feature training data
    - X_val is validation data
    - y_train is the resultant column
    - y_val is the result validation data
    """)
    return


@app.cell
def _(X, train_test_split, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    # print the shapes
    print(f"X original shape: ", {X.shape})
    print(f"X_train shape: ", {X_train.shape})
    print(f"X_val shape: ", {X_val.shape})
    print(f"y original shape: ", {y.shape})
    print(f"y_train shape: ", {y_train.shape})
    print(f"y_val shape: ", {y_val.shape})
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As Sean's EDA showed, there are 3 classes of data:
    - Numerical
    - Categorical
    - Ordinal

    We create lists for each type of data as the preprocessor would have to process each class of data differently
    """)
    return


@app.cell
def _(X, np):
    # best approach is to use a dictionary for the ordinal categories wherein the
    # key is the column name and the value is a list of the categories
    ordinal_map = {
        'sleep_quality': ["poor", "average", "good"],
        "physical_activity_level": ["sedentary", "moderate", "active"],
        "stress_level": ["low", "medium", "high"]
    }

    ordinal_cols = list(ordinal_map.keys())
    ordinal_categories = list(ordinal_map.values())

    # split data between categorical, ordinal and numerical data
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # This was found out from the EDA previously done by Sean
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # remove the ordinal columns
    categorical_cols = [col for col in categorical_cols if col not in ordinal_cols]
    print(f"numerical_cols {numerical_cols}")
    print(f"ordinal_cols {ordinal_cols}")
    print(f"categorical cols {categorical_cols}")
    print(f"Ordinal categories {ordinal_categories}")
    return categorical_cols, numerical_cols, ordinal_categories, ordinal_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create the preprocessor pipeline that will handle missing value imputation, encoding, and scaling of data
    """)
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
):
    preprocessor = ColumnTransformer(
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
    return (preprocessor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the preprocessor on the data
    """)
    return


@app.cell
def _(X_test, X_train, X_val, preprocessor):
    # use the preprocessor on the datasets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    return X_test_processed, X_train_processed, X_val_processed


@app.cell
def _(categorical_cols, numerical_cols, ordinal_cols, preprocessor):
    ord_feature_names = preprocessor.named_transformers_['ord'].named_steps['encoder'].get_feature_names_out(ordinal_cols)
    num_feature_names = numerical_cols
    indicator_names = [f"{col}_missing" for col in numerical_cols]
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)

    print(ord_feature_names)
    all_feature_names = list(num_feature_names) + list(indicator_names) + list(ord_feature_names) + list(cat_feature_names)

    print(f"Feature count {len(all_feature_names)}")
    print(f"Feature names {all_feature_names}")
    return (all_feature_names,)


@app.cell
def _(X_train_processed, X_val_processed, all_feature_names, pd):
    # turn the processed data into dataframes
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_val_processed_df = pd.DataFrame(X_val_processed, columns=all_feature_names)
    X_test_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)

    print(f"X_train_processed dataframe {X_train_processed_df.shape}")
    print(f"X_val_processed dataframe {X_val_processed_df.shape}")
    print(f"X_test_processed dataframe {X_test_processed_df.shape}")
    return (X_train_processed_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These next cells are just to view the preprocessed data
    """)
    return


@app.cell
def _(X_test_processed, X_train_processed, X_val_processed, np):
    # checking if preprocessed data contains any NaNs
    nan_count_train = np.isnan(X_train_processed).sum()
    nan_count_val = np.isnan(X_val_processed).sum()
    nan_count_test = np.isnan(X_test_processed).sum()

    print(f"nan count in X_train_processed is {nan_count_train}")
    print(f"nan count in X_val_processed is {nan_count_val}")
    print(f"nan count in X_test is {nan_count_test}")
    return


@app.cell
def _(X_train_processed_df, pd):
    # print every column in the X_train_processed dataframe
    pd.set_option("display.max_columns", None)
    X_train_processed_df.head()
    return


@app.cell
def _(X_train_processed_df, numerical_cols):
    # check medians and IQRs after scaling
    X_train_processed_df[numerical_cols].describe().loc[["mean", "std", "50%"]]
    return


@app.cell
def _(train_data_1, y_train):
    print(y_train.value_counts(normalize=True))
    print(train_data_1['health_condition'].value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Encode the y_train data which contains only the health_condition from the train_data
    """)
    return


@app.cell
def _(LabelEncoder, y_train, y_val):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    print(label_encoder.classes_)
    return


if __name__ == "__main__":
    app.run()
