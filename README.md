# Micromobility Weather Sensitivity in Los Angeles

Hourly Metro Bike ridership, matched to weather, modeled as a panel of regions over time.

This repo builds a reproducible pipeline from raw trips and API weather to fixed effects count models, temporal validation, and clear takeaways about how rain and temperature shape bikeshare demand in Los Angeles.

---

## Project goals

1. Assemble a clean region by hour panel of Metro Bike trips joined to Open Meteo weather.
2. Estimate within hour sensitivity of ridership to temperature, rain, and weekends, controlling for region, month, and time of day.
3. Use Poisson and Negative Binomial GLM with fixed effects and clustered standard errors rather than naive regression.
4. Run temporal validation and compare models with and without temperature to see what really matters for prediction.
5. Build something that can be reused for a New York City extension.

---

## High level summary

- **Scope**  
  - Los Angeles Metro Bike share  
  - Three regions: Downtown LA, Westside, North Hollywood  
  - Hourly panel, region by date from mid 2024 through 2025, plus a train and validation split

- **Methods**  
  - Region by hour aggregation with Haversine distance and duration summaries  
  - OLS with fixed effects as a baseline  
  - Poisson GLM with overdispersion checks and Negative Binomial refit  
  - Cluster robust standard errors by day  
  - Temporal validation: train on earlier period, test on later period  
  - Model comparison of specifications with and without temperature

- **Headline results (Los Angeles)**  
  - Strong diurnal pattern: peak afternoon hours have roughly one order of magnitude more trips than late night hours, after controls.  
  - Rain matters a lot: about a 27 percent drop in hourly trips per millimeter of rain.  
  - Weekends are modestly busier: about 8 percent more trips than weekdays, controlling for hour and month.  
  - Temperature has a small within hour effect: roughly a 0.7 percent decrease in trips per degree Celsius, and it adds almost no predictive power once month and hour fixed effects are in the model.

---

## Repository structure

Core pieces, starting from raw data and moving toward models:

```text
assembly/
  clean_metro_trips.py     # Clean and filter raw Metro Bike trip CSVs
  acquire_weather_data.py  # Pull hourly Open Meteo weather for each region
  assemble_panels.py       # Join trips + weather into hourly panels
  constants.py             # Date windows and tags (train vs validate), to be mutated by user

data/
  raw/                     # Quarterly Metro Bike trips, station metadata
  processed/
    metro_trips/           # Cleaned trip files by tag
    weather_data/          # Region specific hourly weather by tag
    panels/
      hourly_train.csv
      hourly_validate.csv
      hourly_24-25.csv     # Modeling panel

models/
  hourly_OLS_model.py      # OLS fixed effects baseline
  hourly_GLM_model.py      # Poisson + NB2 GLM with fixed effects

validation/
  hourly_temporal_validation.py    # Temporal train vs validate
  model_with_vs_without_temperature.py  # Temperature spec comparison

results/
  OLS/
    OLS_regression_summary.txt
    OLS_elasticity_table.csv
  GLM/
    GLM_regression_summary_Poisson.txt
    GLM_regression_summary_NB2.txt
    GLM_elasticity_table.csv
    GLM_temperature_comparison.txt
  validation/
    validation_metrics_by_spec.txt
    validation_metrics.txt

plots/
  OLS/
    predicted_vs_actual_hourly.png
  GLM/
    predicted_vs_actual_hourly_NB2.png
  validation/
    predicted_vs_actual_validation_FULL.png
    predicted_vs_actual_validation_REDUCED.png
  assembly/
    hourly_trips_by_region.png
    temp_vs_trips_scatter.png
