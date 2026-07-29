import marimo

__generated_by = "0.10.0"
app = marimo.App(
    width="medium",
    app_title="Linear & Logistic Regression Interactive Explorer",
)


@app.cell
def _():
  import marimo as mo
  import numpy as np
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
  from sklearn.metrics import accuracy_score, mean_squared_error
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import PolynomialFeatures

  return (
      mo,
      np,
      go,
      make_subplots,
      LinearRegression,
      Ridge,
      Lasso,
      LogisticRegression,
      accuracy_score,
      mean_squared_error,
      make_pipeline,
      PolynomialFeatures,
  )


@app.cell
def _(mo):
  mo.md(
      r"""
    # 📈 Interactive Linear & Logistic Regression Explorer

    This interactive notebook demonstrates how hyper-parameters and data properties affect 
    **Linear Regression** and **Logistic Regression** models. Adjust the UI controls below to observe 
    changes in model fit, weight distribution, and loss behavior.
    """
  )
  return


@app.cell
def _(mo):
  # Global controls
  model_type = mo.ui.dropdown(
      options=["Linear Regression", "Logistic Regression"],
      value="Linear Regression",
      label="Model Type",
  )

  n_samples = mo.ui.slider(
      start=20, stop=300, step=10, value=80, label="Sample Size (N)"
  )

  noise_level = mo.ui.slider(
      start=0.0,
      stop=3.0,
      step=0.2,
      value=1.0,
      label="Noise / Class Overlap Level",
  )

  poly_degree = mo.ui.slider(
      start=1, stop=10, step=1, value=1, label="Polynomial Degree"
  )

  reg_type = mo.ui.dropdown(
      options=["None", "L2 (Ridge)", "L1 (Lasso)"],
      value="None",
      label="Regularization",
  )

  reg_alpha = mo.ui.slider(
      start=0.001,
      stop=100.0,
      step=0.5,
      value=1.0,
      label="Regularization Strength (α / C⁻¹)",
  )

  add_outliers = mo.ui.checkbox(value=False, label="Inject Outliers")

  mo.hstack([
      mo.vstack([model_type, n_samples, noise_level, add_outliers]),
      mo.vstack([poly_degree, reg_type, reg_alpha]),
  ])
  return (
      add_outliers,
      model_type,
      n_samples,
      noise_level,
      poly_degree,
      reg_alpha,
      reg_type,
  )


@app.cell
def _(
    LinearRegression,
    Lasso,
    LogisticRegression,
    PolynomialFeatures,
    Ridge,
    accuracy_score,
    add_outliers,
    go,
    make_pipeline,
    make_subplots,
    mean_squared_error,
    model_type,
    n_samples,
    noise_level,
    np,
    poly_degree,
    reg_alpha,
    reg_type,
):
  # Generate reproducible dataset based on UI state
  np.random.seed(42)
  N = n_samples.value
  noise = noise_level.value
  degree = poly_degree.value
  alpha = reg_alpha.value

  # Create plot grid
  fig = make_subplots(
      rows=1,
      cols=2,
      subplot_titles=(
          "Model Fit & Decision Space",
          "Model Coefficients (Weights)",
      ),
  )

  if model_type.value == "Linear Regression":
    # --- 1D LINEAR REGRESSION DATASET ---
    X = np.linspace(-3, 3, N).reshape(-1, 1)
    y_true = 0.5 * X.squeeze() ** 2 - X.squeeze() + 1
    y = y_true + np.random.normal(0, noise, size=N)

    if add_outliers.value:
      # Inject leverage outliers
      outlier_idx = np.random.choice(N, size=int(N * 0.08), replace=False)
      y[outlier_idx] += np.random.choice([-15, 15], size=len(outlier_idx))

    # Helper function for estimator instantiation
    if reg_type.value == "L2 (Ridge)":
      base_model = Ridge(alpha=alpha)
    elif reg_type.value == "L1 (Lasso)":
      base_model = Lasso(alpha=alpha, max_iter=5000)
    else:
      base_model = LinearRegression()

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), base_model
    )
    model.fit(X, y)

    X_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    y_pred = model.predict(X_plot)
    y_train_pred = model.predict(X)

    mse = mean_squared_error(y, y_train_pred)
    weights = model.named_steps[
        base_model.__class__.__name__.lower()
    ].coef_.flatten()

    # Data Plot
    fig.add_trace(
        go.Scatter(
            x=X.squeeze(),
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="#1f77b4", opacity=0.7),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X_plot.squeeze(),
            y=y_pred,
            mode="lines",
            name=f"Fit (Degree {degree})",
            line=dict(color="#ff7f0e", width=3),
        ),
        row=1,
        col=1,
    )

    metric_text = f"**Mean Squared Error (MSE):** `{mse:.4f}`"

  else:
    # --- 2D LOGISTIC REGRESSION DATASET ---
    mean0, mean1 = [-1, -1], [1, 1]
    cov = [[1, 0.3 * noise], [0.3 * noise, 1]]

    X0 = np.random.multivariate_normal(mean0, cov, N // 2)
    X1 = np.random.multivariate_normal(mean1, cov, N // 2)
    X = np.vstack([X0, X1])
    y = np.array([0] * (N // 2) + [1] * (N // 2))

    if add_outliers.value:
      # Inject mislabeled noise points
      flip_idx = np.random.choice(N, size=int(N * 0.10), replace=False)
      y[flip_idx] = 1 - y[flip_idx]

    penalty = (
        "l2"
        if reg_type.value == "L2 (Ridge)"
        else ("l1" if reg_type.value == "L1 (Lasso)" else None)
    )
    C_val = 1.0 / alpha if penalty is not None else 1.0

    solver = "saga" if penalty == "l1" else "lbfgs"
    base_model = LogisticRegression(
        penalty=penalty, C=C_val, solver=solver, max_iter=2000
    )

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False), base_model
    )
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))
    weights = base_model.coef_.flatten()

    # Decision Boundary Mesh Grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Contour Plot for Probabilities
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 150),
            y=np.linspace(y_min, y_max, 150),
            z=Z,
            colorscale="RdBu",
            reversescale=True,
            opacity=0.4,
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Scatter points by class
    fig.add_trace(
        go.Scatter(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            mode="markers",
            name="Class 0",
            marker=dict(color="blue", symbol="circle"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            name="Class 1",
            marker=dict(color="red", symbol="x"),
        ),
        row=1,
        col=1,
    )

    metric_text = f"**Classification Accuracy:** `{acc * 100:.2f}%`"

  # Weights bar chart
  feature_names = [f"w_{i + 1}" for i in range(len(weights))]
  fig.add_trace(
      go.Bar(
          x=feature_names,
          y=weights,
          name="Weights",
          marker_color=[
              "#2ca02c" if abs(w) > 0.001 else "#d62728" for w in weights
          ],
      ),
      row=1,
      col=2,
  )

  fig.update_layout(
      height=480,
      margin=dict(l=20, r=20, t=40, b=20),
      template="plotly_white",
  )

  return fig, metric_text


@app.cell
def _(fig, metric_text, mo):
  mo.vstack([mo.md(metric_text), mo.ui.plotly(fig)])
  return


@app.cell
def _(mo):
  mo.md(
      r"""
    ---
    ### 💡 Key Learning Takeaways

    * **Polynomial Expansion & Overfitting:** Increasing the degree allows the model to capture non-linear patterns, but high degrees without regularization lead to severe **overfitting** (extreme wild curves and giant weight values).
    * **L2 Regularization (Ridge):** Shrinks feature weights toward zero smoothly, reducing variance without driving coefficients completely to zero.
    * **L1 Regularization (Lasso):** Forces redundant or insignificant feature weights strictly to **0**, performing implicit feature selection.
    * **Outlier Sensitivity:** Linear models fitted with Mean Squared Error (MSE) or standard Log Loss can be heavily skewed by high-leverage outliers.
    """
  )
  return


if __name__ == "__main__":
  app.run()
