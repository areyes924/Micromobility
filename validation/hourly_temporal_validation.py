import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import statsmodels.discrete.discrete_model as dm
import matplotlib.pyplot as plt

"""

Hourly GLM Temporal Validation

Evaluates out-of-sample predictive performance using a temporal train/validate split.

Much of the logic is the same as it is in hourly_GLM_model.py

Outputs:
- Text report with metrics by spec:
    results/validation/validation_metrics_by_spec.txt
- Predicted vs actual scatterplots on the validation set:
    plots/validation/predicted_vs_actual_validation_FULL.png
    plots/validation/predicted_vs_actual_validation_REDUCED.png

Interpretation:
- Compare FULL vs REDUCED to see whether including temperature_c
  improves temporal generalization beyond rain and fixed effects.
- Higher predictive R^2 and lower RMSE or bias for FULL indicate that
  temperature adds useful signal for forecasting hourly trips over time.
"""

# ======================
# Setup constants
# ======================

TRAIN_PATH = "data/processed/panels/hourly_train.csv"
VALIDATE_PATH = "data/processed/panels/hourly_validate.csv"

OUT_METRICS = "results/validation/validation_metrics_by_spec.txt"
OUT_PLOT_FULL = "plots/validation/predicted_vs_actual_validation_FULL.png"
OUT_PLOT_REDUCED = "plots/validation/predicted_vs_actual_validation_REDUCED.png"

DISPERSION_SWITCH = 1.5

# ======================
# Load + Clean Data
# ======================

train = pd.read_csv(TRAIN_PATH)
valid = pd.read_csv(VALIDATE_PATH)

train["date"] = pd.to_datetime(train["date"])
valid["date"] = pd.to_datetime(valid["date"])

train = train[train["missing_weather_flag"] == 0].copy()
valid = valid[valid["missing_weather_flag"] == 0].copy()

train["cluster_day"] = train["date_id"].astype(int)
valid["cluster_day"] = valid["date_id"].astype(int)

specs = { # Testing with and without temperature_c
    "FULL": (
        "trip_count ~ temperature_c + rain_mm + weekend_flag "
        "+ C(region) + C(month) + C(hour)"
    ),
    "REDUCED": (
        "trip_count ~ rain_mm + weekend_flag "
        "+ C(region) + C(month) + C(hour)"
    ),
}

lines = []

# ======================
# Run model for each spec
# ======================

for label, formula in specs.items():
    # design matrices on train
    y_train, X_train = patsy.dmatrices(formula, train, return_type="dataframe")

    # fit poisson model
    glm = sm.GLM(y_train, X_train, family=sm.families.Poisson())
    res = glm.fit(cov_type="cluster", cov_kwds={"groups": train["cluster_day"]})

    # dispersion check, switch to nb2 if necessary
    disp = res.pearson_chi2 / res.df_resid
    if disp > DISPERSION_SWITCH:
        nb_mod = dm.NegativeBinomial(y_train, X_train)
        nb_res = nb_mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": train["cluster_day"]},
            maxiter=200,
            disp=False,
        )
        final_model = nb_res
        model_used = "NB2"
    else:
        final_model = res
        model_used = "Poisson"

    # align validation design matrix to train columns
    train_cols = X_train.columns
    y_val, X_val_raw = patsy.dmatrices(formula, valid, return_type="dataframe")
    X_val = X_val_raw.reindex(columns=train_cols, fill_value=0.0)

    # Predict on validation set
    valid[f"predicted_{label}"] = np.asarray(final_model.predict(X_val)).ravel()

    y_true = valid["trip_count"].to_numpy(dtype=float)
    y_pred = valid[f"predicted_{label}"].to_numpy(dtype=float)

    # predictive metrics
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / y_true.shape[0]))
    bias = float((y_pred - y_true).mean())

    lines.append(
        f"{label} model ({model_used})\n"
        f"  Predictive R2: {r2:.4f}\n"
        f"  RMSE: {rmse:.4f}\n"
        f"  Mean bias: {bias:.4f}\n"
    )

    # plots, one file per spec, predicted vs actual
    out_plot = OUT_PLOT_FULL if label == "FULL" else OUT_PLOT_REDUCED

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_true, alpha=0.3, s=10)
    mx = float(max(y_pred.max(), y_true.max()))
    plt.plot([0, mx], [0, mx], "r--", lw=2)
    plt.xlabel(f"Predicted trips (validation, {label})")
    plt.ylabel("Actual trips (validation)")
    plt.title(f"Predicted vs Actual, Validation Set ({label} spec)")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()

# write conclusions
with open(OUT_METRICS, "w") as f:
    f.write("\n".join(lines))
