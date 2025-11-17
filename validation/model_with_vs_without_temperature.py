import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
import statsmodels.discrete.discrete_model as dm

"""
Hourly GLM Temperature Spec Comparison

Compares full vs reduced specifications for the hourly GLM, using
Poisson if dispersion is near 1 and NB2 if overdispersed.

1) FULL:    trip_count ~ temperature_c + rain_mm + weekend_flag
            + C(region) + C(month) + C(hour)

2) REDUCED: trip_count ~ rain_mm + weekend_flag
            + C(region) + C(month) + C(hour)

(Basically, comparing if temperature actually matters for the model)

Outputs LL, AIC, BIC, pseudo R2 for full vs reduced specs.
"""

# ======================
# Define constants
# ======================

HOURLY_PATH = "data/processed/panels/hourly_24-25.csv"
OUT_PATH = "results/GLM/GLM_temperature_comparison.txt"
DISPERSION_SWITCH = 1.5

# ======================
# Load + clean data
# ======================

hourly = pd.read_csv(HOURLY_PATH)

hourly["date"] = pd.to_datetime(hourly["date"], errors="coerce")
hourly_model = hourly.loc[hourly["missing_weather_flag"] == 0].copy()

# cluster by day
hourly_model["cluster_day"] = hourly_model["date_id"].astype(int)

# ======================
# outline each spec
# ======================

formula_full = (
    "trip_count ~ temperature_c + rain_mm + weekend_flag "
    "+ C(region) + C(month) + C(hour)"
)

formula_reduced = (
    "trip_count ~ rain_mm + weekend_flag "
    "+ C(region) + C(month) + C(hour)"
)

formula_null = "trip_count ~ 1"

# ======================
# Dispersion check on Poisson FULL
# ======================

pois_full = smf.glm(
    formula=formula_full,
    data=hourly_model,
    family=sm.families.Poisson()
)
pois_full_res = pois_full.fit(
    cov_type="cluster",
    cov_kwds={"groups": hourly_model["cluster_day"]}
)

dispersion = pois_full_res.pearson_chi2 / pois_full_res.df_resid
print(f"Pearson dispersion (Poisson FULL) = {dispersion:.2f}")

use_nb2 = dispersion > DISPERSION_SWITCH
model_family = "NB2" if use_nb2 else "Poisson"

# ======================
# Build design matrices if we need NB2 (we have every time i have run this)
# ======================

if use_nb2:
    # design matrices for NB2 fits
    y_full, X_full = patsy.dmatrices(formula_full, hourly_model, return_type="dataframe")
    y_red, X_red = patsy.dmatrices(formula_reduced, hourly_model, return_type="dataframe")
    y_null, X_null = patsy.dmatrices(formula_null, hourly_model, return_type="dataframe")

    nb_full = dm.NegativeBinomial(y_full, X_full)
    nb_full_res = nb_full.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]},
        maxiter=200,
        disp=False
    )

    nb_red = dm.NegativeBinomial(y_red, X_red)
    nb_red_res = nb_red.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]},
        maxiter=200,
        disp=False
    )

    nb_null = dm.NegativeBinomial(y_null, X_null)
    nb_null_res = nb_null.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]},
        maxiter=200,
        disp=False
    )

    ll_null = nb_null_res.llf

    results = {
        "FULL": nb_full_res,
        "REDUCED": nb_red_res,
    }

else:
    # stay in Poisson for all three
    null_mod = smf.glm(
        formula=formula_null,
        data=hourly_model,
        family=sm.families.Poisson()
    )
    null_res = null_mod.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]}
    )
    ll_null = null_res.llf

    full_res = pois_full_res  # already fitted above

    red_mod = smf.glm(
        formula=formula_reduced,
        data=hourly_model,
        family=sm.families.Poisson()
    )
    red_res = red_mod.fit(
        cov_type="cluster",
        cov_kwds={"groups": hourly_model["cluster_day"]}
    )

    results = {
        "FULL": full_res,
        "REDUCED": red_res,
    }

# ======================
# Collect metrics
# ======================

summary = {}

for label in ["FULL", "REDUCED"]:
    res = results[label]
    llf = res.llf
    aic = res.aic
    bic = res.bic if hasattr(res, "bic") else np.nan
    pseudo_r2 = 1.0 - llf / ll_null

    summary[label] = {
        "llf": llf,
        "aic": aic,
        "bic": bic,
        "pseudo_r2": pseudo_r2,
    }

# ======================
# Step 4: write summary out
# ======================

lines = []
lines.append(f"Family used for comparison: {model_family}")
lines.append(f"Pearson dispersion (Poisson FULL): {dispersion:.3f}")
lines.append("")

for label in ["FULL", "REDUCED"]:
    r = summary[label]
    lines.append(f"{label} model ({model_family}, clustered SEs)")
    lines.append(f"  Log-likelihood: {r['llf']:.3f}")
    lines.append(f"  AIC:            {r['aic']:.3f}")
    lines.append(f"  BIC:            {r['bic']:.3f}")
    lines.append(f"  Pseudo R2:      {r['pseudo_r2']:.4f}")
    lines.append("")

with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines))

print("Temperature spec comparison written to:", OUT_PATH)
