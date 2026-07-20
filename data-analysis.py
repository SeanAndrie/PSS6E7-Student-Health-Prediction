import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    import os
    import kagglehub
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    return kagglehub, mo, np, pd, plt, sns


@app.cell
def _(kagglehub, pd, sns):
    # kagglehub.login()

    try: 
        path = kagglehub.competition_download('playground-series-s6e7')
    except Exception as e: 
        print("Failed to download competition data. Manually download the dataset from the competitions page.")
    else:
        print(f"-- Downloaded competition data to '{path}' -- ")

    main_dir = "playground-series-s6e7"

    try:
        train_data = pd.read_csv(f"{main_dir}/train.csv")
        test_data = pd.read_csv(f"{main_dir}/test.csv")
        sample_sub = pd.read_csv(f"{main_dir}/sample_submission.csv")
    except FileNotFoundError:
        print(f"Failed to load data. Ensure that '{main_dir}' exists in your current working directory.")
    else: 
        print(f"-- Loaded '{main_dir}' successfully --")

    sns.set_theme(style="darkgrid", palette="Set2")
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell
def _(train_data):
    train_data.info()
    return


@app.cell
def _(train_data):
    train_data.describe()
    return


@app.cell
def _(test_data):
    test_data.head()
    return


@app.cell
def _(test_data):
    test_data.describe()
    return


@app.cell
def _(train_data):
    train_cat = train_data.select_dtypes(exclude="number")
    train_num = train_data.select_dtypes(include="number")

    cat_cols = train_cat.columns.tolist()
    num_cols = train_num.columns.tolist()[1:]

    print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical Columns ({len(num_cols)}): ", num_cols)
    return (num_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Feature information

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
    ## 2. Target Analysis

    ### 2.1. Target Distribution
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: Severe class-imbalance. The `at-risk` class dominates the target with about 85.9% observations compared to the `unhealthy` and `fit` classes at 8.4% and 5.8% respectively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2. Feature distributions by target
    """)
    return


@app.cell
def _(train_data):
    # Create a copy of the training data with the `id` column dropped
    data_no_id = train_data.drop("id", axis=1)
    return


@app.cell
def _(num_cols, pd, plt, sns, train_data):
    def plot_distributions_by_target(
        df:pd.DataFrame, 
        target:str,
        nfeats:list[str], 
        ncols:int, 
        figsize=(15, 15)
    ) -> None:
        nrows = -(-len(nfeats) // ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for i, feat in enumerate(nfeats):
            row, col = i // ncols, i % ncols 
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_title(f"{target} vs. {feat}")
            sns.boxplot(df, x=target, y=feat, hue=target, ax=ax)

        if len(nfeats) % ncols != 0:
            for i in range(len(nfeats) % ncols, ncols):
                axes[-1, i].axis('off')

        fig.tight_layout()

    plot_distributions_by_target(train_data, "health_condition", num_cols, 3)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**:

    **Strong, clearly ordered relationships**:
    - `sleep_duration`: Very clean separation. The boxes barely overlap.
    - `step_count`: Fit stands well apart from unhealthy/at-risk, which overlap heavily with each other.
    - `exercise_duration`: Similar pattern to `step_count`. Fit is clearly higher and tighter, while unhealthy and at-risk overlap substantially.
    - `bmi`: Monotonic and ordered. Though the boxes overlap quite a bit more than `sleep_duration`, it is still directionally informative.

    **Weak or negligible relationships**:
    - `heart_rate`: The three boxes are nearly identical.
    - `calorie_expenditure`: Medians and IQRs are very close across all three classes with heavy overlap.
    - `water_intake`: Essentially identical distributions across all three groups.

    **Fit** is the most distinct class, especially on `step_count` and `exercise_duration`. It's clearly separated from the two. **Unhealthy** vs **At-Risk** is the harder distinction since for several features (`step_count`, `calorie_expenditure`, `heart_rate`) their boxes overlap almost completely. This suggests that the model may struggle most to distinguish these two classes from each other.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3. Target Correlation
    """)
    return


@app.cell
def _(np, pd, plt, sns, train_data):
    def plot_target_correlation(
        data: pd.DataFrame,
        target: str,
        target_map:dict,
        fmt=".2f",
        cmap="Greens",
        figsize=(18, 12)
    ) -> None:
        copy = data.select_dtypes(include="number").drop("id", axis=1)
        copy[target] = data[target].map(target_map)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 3)

        corr_hm_ax = fig.add_subplot(gs[0, :2])
        corr_target_ax = fig.add_subplot(gs[0, 2])
        mask = np.tril(copy.corr())

        corr_hm_ax.set_title("Pearson Correlation Matrix", size=15)
        corr_target_ax.set_title(f"Correlation to Target ({target})", size=15)

        sns.heatmap(copy.corr(), mask=mask, annot=True, fmt=fmt,
                    cmap=cmap, ax=corr_hm_ax, cbar=False, annot_kws={"size": 10})

        sns.heatmap(copy.corr()[target].sort_values(ascending=False).to_frame(),
                    annot=True, fmt=fmt, cmap=cmap, ax=corr_target_ax)

        corr_target_ax.set_yticklabels(corr_target_ax.get_yticklabels(), rotation=0)
        corr_hm_ax.set_xticklabels(corr_hm_ax.get_xticklabels(), rotation=45)

    plot_target_correlation(train_data, "health_condition", {"unhealthy": 0, "at-risk": 1, "fit": 2})
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: Confirms some previous observations from viewing the feature distributions from the target.

    - `sleep_duration` and `bmi` show strong linear signals. This is consistent with the clean, monotonic separation in the box plots.
    - `heart_rate` and `water_intake` are essentially uncorrelated, matching the fully-overlapping boxplots.
    - `step_count` and `exercise_duration` show up as a weak-ish signal here, which *understates* their real value given what the boxplots showed (strong for separating "fit", weak for the other two classes).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Missing Data
    """)
    return


@app.cell
def _(pd, plt, train_data):
    def plot_missing_proportion(df:pd.DataFrame, subset:str, figsize=(15, 10)) -> None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        total = df.shape[0]
        missing = df.isna().sum().sum()
    
        axes[0].pie(x=[total, missing], labels=["Overall", "Missing"], autopct="%1.1f%%")
        axes[0].set_title("Overall vs Missing Values")

        missing_pct = df.drop("id", axis=1).isna().mean() * 100
        present_pct = 100 - missing_pct 
    
        p = axes[1].barh(missing_pct.index, missing_pct, color="#fc8d62", label="Present") #66c2a5
        axes[1].barh(missing_pct.index, present_pct, left=missing_pct, color="#66c2a5",label="Missing") #fc8d62
        axes[1].set_title("Missing values per feature (%)")
        axes[1].set_xlabel("Percentage (%)")
        axes[1].set_xlim(0, 100)
        axes[1].legend(loc="lower right")
        axes[1].bar_label(p, label_type="edge", fmt="%1.1f%%", padding=4)

        fig.suptitle(f"Missing Data ({subset})")
        fig.tight_layout()

    plot_missing_proportion(train_data, "Train")
    plt.gca()
    return (plot_missing_proportion,)


@app.cell
def _(plot_missing_proportion, plt, test_data):
    plot_missing_proportion(test_data, "Test")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**:
    - Nearly half of the dataset consists of missing values with about **39.4%** missing and **60.6%** overall.
    - Missingness is low in absolute terms, but scattered across nearly every feature.
    - There's a clear split between subjective/self-reported fields and objective/logged fields.

    Looking at which features have high missing rates:

    - `stress_level`, `sleep_duration`, `sleep_quality`, `water_intake`, `calorie_expenditure`.

    versus the lowest:

    - `diet_type`, `exercise_duration`, `bmi`, `step_count`, `heart_rate`.

    The high-missing group tends to be things people self-report or that require wearable/consistent tracking, while the low-missing group looks more like fields that are either directly measured or mandatory intake questions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train and Test data distributions

    ### 4.1. Numerical features
    """)
    return


@app.cell
def _(pd, plt, sns, test_data, train_data):
    def plot_num_feat_distributions(
        train_data:pd.DataFrame, 
        test_data:pd.DataFrame, 
        ncols:int,
        bins=30,
        figsize=(15, 15)
    ) -> None:
        train_num = train_data.select_dtypes(include="number")
        test_num = test_data.select_dtypes(include="number") 

        nfeats = train_num.columns.tolist()[1:]
        nrows = -(-len(nfeats ) // ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for i, feat in enumerate(nfeats):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            sns.histplot(data=train_num, x=feat, bins=bins, kde=True, ax=ax)
            sns.histplot(data=test_num, x=feat, bins=bins, kde=True, ax=ax)
            ax.set_title(f"{feat}")

        if len(nfeats) % ncols != 0: 
            for i in range(len(nfeats) % ncols, ncols):
                axes[-1, i].axis('off')

        fig.tight_layout()

    plot_num_feat_distributions(train_data, test_data, 2)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: No visible distribution drift between train and test data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2. Categorical and Ordinal features

    **TODO**: Account for ordinal mapping and visualize proportion instead of raw counts.
    """)
    return


@app.cell
def _(pd, plt, sns, test_data, train_data):
    def plot_cat_feat_distributions(
        train_data:pd.DataFrame,
        test_data:pd.DataFrame,
        target:str,
        # ordinal_map: dict[str, list[str]] = None, 
        figsize=(12, 25)
    ) -> None:
        train_cat = train_data.select_dtypes(exclude="number").drop(target, axis=1)
        test_cat = test_data.select_dtypes(exclude="number")
        nfeats = train_cat.columns.tolist()
        fig, axes = plt.subplots(nrows=len(nfeats), ncols=2, figsize=figsize)
    
        for i, feat in enumerate(nfeats):
            sns.countplot(train_data, x=feat, hue=feat, ax=axes[i, 0])
            sns.countplot(test_data, x=feat, hue=feat, ax=axes[i, 1])

            axes[i, 0].set_title("Train")
            axes[i, 1].set_title("Test")
        
        fig.tight_layout()

    plot_cat_feat_distributions(train_data, test_data, "health_condition")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
