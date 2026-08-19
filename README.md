# PS-S6E7 | Predicting Student Health Risk: Workshop Walkthrough

This is a walkthrough of a solution of the Playground Series Competition for the AI/ML Club Workshop on "Applied ML & Predictive Modelling" of 42 Abu Dhabi.

## Competition Overview

**Playground Series Season 6, Episode 7**: Predicting Student Health Risk

The dataset contains 50,000 records of college students' lifestyle, physiological, and psychological characteristics, with the goal of classifying students into three health categories: `Fit`, `At-Risk`, or `Unhealthy`.

## Workshop Topics

### 1. Exploratory Data Analysis (EDA)
- Target distribution analysis and class imbalance identification
- Numerical feature distributions by target class
- Correlation analysis between features and target
- Missing data patterns and their implications

### 2. Evaluation Metrics for Imbalanced Classification
- **Balanced Accuracy**: Implementation from scratch and comparison with scikit-learn
- Understanding why plain accuracy fails with imbalanced datasets
- Macro-averaged recall concept and its importance

### 3. Cross-Validation Strategies
- Limitations of single train-test splits
- Regular K-Fold vs Stratified K-Fold cross-validation
- Impact of stratification on class distribution across folds

### 4. Model Training and Evaluation
- **Baseline Models**: Logistic Regression, Decision Trees
- **Ensemble Methods**: Random Forest, HistGradientBoosting
- **Gradient Boosting**: XGBoost, LightGBM
- Class weighting techniques for handling imbalance

### 5. Feature Engineering
- Ordinal encoding for ranked categorical features
- One-hot encoding for nominal categories
- Feature preprocessing pipeline construction

### 6. Model Interpretation
- SHAP (SHapley Additive exPlanations) values
- Feature importance analysis
- Permutation importance calculation

### 7. End-to-End ML Pipeline
- Data loading and preprocessing
- Model training with cross-validation
- Prediction generation and submission

## Key Learning Objectives

By the end of this workshop, participants will be able to:

1. Choose appropriate evaluation metrics for imbalanced classification problems
2. Implement balanced accuracy from scratch to understand how it differs from plain accuracy
3. Apply stratified K-Fold cross-validation to get reliable performance estimates with imbalanced classes
4. Use class weighting to handle severe class imbalance during model training
5. Compare model families under controlled conditions to identify the best performer
6. Engineer features to expose signal hidden in the raw representation
7. Interpret model predictions using SHAP values to understand feature importance
8. Build a complete ML pipeline from EDA through submission

## Dependencies

- `uv` - Package installer for Python
- `kagglehub` - For dataset download
- `numpy`, `pandas` - Data manipulation
- `seaborn`, `matplotlib` - Visualization
- `scikit-learn` - ML utilities and models
- `xgboost`, `lightgbm` - Gradient boosting frameworks
- `shap` - Model interpretability
- `jupyter` - Jupyter notebook support (optional)

## Running the Walkthrough

1. Install dependencies:
   ```bash
   uv pip install kagglehub numpy pandas seaborn matplotlib scikit-learn xgboost lightgbm shap
   ```

2. Download the dataset (requires Kaggle API token):
   ```python
   import kagglehub
   kagglehub.login()
   kagglehub.competition_download('playground-series-s6e7')
   ```

3. Run the walkthrough:

   **Option A: Marimo Notebook (Interactive Python)**
   ```bash
   # To edit the notebook interactively:
   uv run --with marimo marimo edit student-health-risk-walkthrough.py

   # To run the notebook as a web app:
   uv run --with marimo marimo run student-health-risk-walkthrough.py
   ```

   **Option B: Jupyter Notebook (Recommended)**
   ```bash
   # Navigate to the notebooks directory:
   cd __marimo__

   # Launch Jupyter and open the notebook:
   jupyter notebook student-health-risk-walkthrough.ipynb
   ```

   **Option C: Google Colab (Cloud-based)**
   ```bash
   # Upload the notebook to Google Colab:
   # 1. Go to colab.research.google.com
   # 2. Click "File" > "Upload notebook"
   # 3. Upload __marimo__/student-health-risk-walkthrough.ipynb
   # 4. Install dependencies in the first cell
   ```

## Workshop Format

This walkthrough is designed as a Jupyter notebook for interactive learning. The workshop includes:

- **Exercises**: Hands-on implementation tasks
- **Visualizations**: Charts and plots for data understanding
- **Comparisons**: Model performance analysis across different approaches
- **Discussion Points**: Key observations and insights from the analysis
