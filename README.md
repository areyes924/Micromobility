# Micromobility Demand and Weather



An hourly panel analysis of bikeshare demand in LA and NYC using fixed-effects count models.



## Project motivation





Urban bikeshare systems generate detailed, high-frequency data, but interpreting it is difficult. Ridership changes sharply by time of day, season, and location, and weather effects depend heavily on context. Without careful modeling, observational data can lead to overstated conclusions.



This project builds a clean hourly panel from large bikeshare datasets and analyzes how ridership varies with short-term weather using interpretable statistical models that make uncertainty and modeling assumptions explicit.



**Research Question:**

**Conditional on time and place, how does bikeshare demand move with weather, and how do these relationships differ between cities?**

Libraries used:
* numpy
* pandas
* matplotlib
* statsmodels
* OpenMeteoAPI



## Modeling approach



### Hourly fixed-effects count models



The primary analysis is implemented in:



model/hourly\_GLM\_model.py



For each city, the model takes the form:

trip\_count ~ weather variables + weekend indicator + C(region) + C(month) + C(hour)



* Poisson GLM fit first; Negative Binomial (NB2) used when overdispersion is present.
* Standard errors are clustered by day (date\_id).
* Uncertainty is reported using clustered standard errors and confidence intervals.
* Predicted values are used only for diagnostics and visualization, not for forecasting claims.



### Specification comparison: with vs. without temperature



I wasn't sure if temperature actually contributed explanatory power once accounting for hour of day/month of year. To test this, I included:



model/GLM\_model\_with\_vs\_without\_temperature.py



This script compares two specifications:



* FULL: includes temperature
* REDUCED: excludes temperature



Models are compared in-sample using:



* Log-likelihood
* AIC
* BIC
* Pseudo R² relative to a null model



These comparisons are used to assess redundancy and specification choice.



## Data sources



In both cases, time windows were chosen during periods of low/no expansion so that wouldn't affect the model's interpretation.



### Los Angeles - Metro Bike Share



* Time window: July 1, 2024 -> September 30, 2025 (1.25 years)
* Scale: ~538k raw trips
* Regions: Downtown LA, Westside, North Hollywood
* Inputs: Quarterly Metro Bike Share trip files and station metadata
* Dataset: https://bikeshare.metro.net/about/data/



Cleaning and filtering (intentionally conservative):



* Drop trips with missing timestamps, stations, or duration
* Remove virtual station 3000
* Drop duplicates on (start time, end time, stations, bike ID)
* Cap duration at 10 hours (to remove system outages and abandoned bikes)
* Remove same-station trips under 15 minutes
* Trips are aggregated to region × date × hour. Zero-trip hours are explicitly preserved by left-joining onto the full weather time grid.



### New York City - Citi Bike



* Time window: December 31, 2022 -> December 31, 2023 (one year)
* Scale: ~35 million raw trips
* Regions: Simple lat/lon geometry buckets (South, Middle, North, East)
* Inputs: Many monthly Citibike trip files (not uploaded to GitHub)
* Dataset: https://citibikenyc.com/system-data



Processing notes:



* Monthly CSVs ingested in chunks to manage scale
* Duration and same-station filters mirror the LA pipeline
* Since there was no station map dataset, region assignment was performed via approximate bounding boxes (using latitude+longitude) -- Expansion of project could find exact latitude/longitude maps of NYC and assign each start trip to a borough. This solution makes region effects arbitrary, but shouldn't affect weather coefficients.
* Final panel structure matches LA (region × date × hour, zero-trip hours preserved) to feed into same model.



### Weather data



* Source: OpenMeteo archive API
* Dataset: https://open-meteo.com/en/docs/historical-weather-api
* Variables used in modeling: temperature, rain, snow (when present)
* Representation: One center-point latitude/longitude per region



Time handling:

* Weather indexed in UTC and converted to local time
* Fall-back DST duplicate hour removed (earlier UTC hour retained)
* Weather generation is standardized and reproducible across all regions and cities.



## Results



### New York City



NYC ridership is highly overdispersed under Poisson (Pearson dispersion ≈ 128.6), so NB2 is used.



Temperature contributes explanatory value even after controlling for region, month, and hour fixed effects (FULL beats REDUCED by ΔAIC ≈ −1287 and ΔBIC ≈ −1278).



Conditional associations in the NB2 model:



* temperature is positive (0.0245 per °C, 95% CI \[0.019, 0.030])
* rain is negative (−0.112 per mm, CI \[−0.148, −0.076])
* snow is strongly negative (−0.83 per mm, CI \[−1.20, −0.46])
* weekend differences are not distinguishable from zero in this specification



### Los Angeles



LA shows mild overdispersion under Poisson (Pearson dispersion ≈ 3.0); NB2 is used for consistency and to account for remaining dispersion.



Temperature improves fit modestly (ΔAIC ≈ −137; ΔBIC ≈ −129) and its estimated association is small and negative (−0.0119 per °C, CI \[−0.017, −0.007]).



* Rain is negative and large in magnitude (−0.33 per mm), though estimates are less precise because rain events are comparatively rare.
* There was no snowfall in any of the LA regions during the timeframe, so snow is excluded from the model
* Weekends are modestly higher in this specification (0.079, about an 8% conditional difference).



### Cross-city interpretation



The modeling framework is consistent across cities, but results differ meaningfully:



* NYC shows stronger weather sensitivity and higher residual variability
* LA shows stronger calendar structure and weaker temperature effects



These differences are city-specific conditional associations.

## 

## What changed during development / what I learned



1 / Different types of models



I was initially using an Ordinary Least Squares (OLS) model on the panel. When plotting predicted versus actual values, I saw that the model produced negative predicted trip counts, which prompted me to notice other issues with OLS. I learned OLS assumes continuous outcomes and constant variance, while hourly trip counts are discrete, non-negative, and very variable. These violations meant that both the predictions and the inferred effects were not interpretable, leading me to switch to count models designed for this setting.



2 / Refining model specification



An early specification included e-bike usage as a covariate. This fed rider behavior back into the model and blurred the interpretation of weather effects. Removing it clarified the estimand and refocused the model on how demand varies with exogenous conditions.



3 / Scaling constraints



Moving from LA to NYC required re-architecting the pipeline around hourly aggregation to handle tens of millions of trips while preserving consistent modeling inputs.





## Repository Structure



```
assembly/
  constants.py
  acquire_weather_data.py
  LA_clean_metro_trips.py
  LA_assemble_panels.py
  NYC_assemble_panels.py

data/
  raw/
    LA/
      metro-bike-share-stations.csv
      metro-trips-*.csv
  processed/
    metro_trips/
      metro_trips_24-25.csv
    weather_data/
      DTLA_hourly_24-25.csv
      Westside_hourly_24-25.csv
      North_Hollywood_hourly_24-25.csv
      NYC_*_hourly_2023.csv
    panels/
      LA_hourly_24-25.csv
      NYC_hourly_2023.csv

model/
  hourly_GLM_model.py
  GLM_model_with_vs_without_temperature.py
  obsolete_hourly_OLS_model.py

results/
  GLM/
    LA/
      GLM_regression_summary_*.txt
      GLM_Full_Vs_Reduced.txt
      GLM_elasticity_table.csv
    NYC/
      GLM_regression_summary_*.txt
      GLM_Full_Vs_Reduced.txt
      GLM_elasticity_table.csv

plots/
  assembly/
    LA_hourly_trips_by_region.png
    NYC_hourly_trips_by_region.png
  GLM/
    LA_predicted_vs_actual_hourly_NB2.png
    NYC_predicted_vs_actual_hourly_NB2.png

```



Raw Citi Bike trip files (~10GB) are not included due to size constraints. The final hourly panel for NYC is included so all modeling results are reproducible.

### 

## Interpretation notes / Summary



These results describe conditional associations, not causal effects. Time and location fixed effects explain most of the variation, so the weather coefficients capture how demand changes within the same place and time context. Differences between Los Angeles and New York City reflect structural differences, as well as modeling choices.



Overall, this project demonstrates an approach for studying short-term weather sensitivity in large urban bikeshare systems using reproducible panel construction and interpretable count models.

