import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import constants

'''
Los Angeles Weather Data Acquisition Script

This script retrieves and formats hourly and daily weather data from the
Open-Meteo Historical API for selected Los Angeles regions during a given time frame.
Outputs are standardized CSVs aligned with micromobility trip datasets.

Notes:
----------------------
- Includes region selector (DTLA, Westside, North Hollywood).
- Handles DST duplicates and missing data flags.
- Adds date, hour, month, and weekend columns for modeling.

Data/Python Skeleton from here:
https://open-meteo.com/en/docs/historical-weather-api?start_date=2024-09-23&end_date=2025-09-22&timezone=America%2FLos_Angeles&hourly=temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,relative_humidity_2m,apparent_temperature,cloud_cover&latitude=34.056278&longitude=-118.231773&daily=temperature_2m_mean,rain_sum,wind_speed_10m_max

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

START = constants.START
END = constants.END
tag = constants.TAG

target = "Westside"

# These are all in the approximate center of the defined regions.
LAT_LONG = {
    "DTLA": (34.05661, -118.237213), # Union Station Coordinates
    "Westside": (34.022449,-118.438332), # Barrington and National Coordinates
    "North Hollywood": (34.16552,-118.375153) # North Hollywood Plaza Coordinates
}

LATITUDE = LAT_LONG[target][0]
LONGITUDE = LAT_LONG[target][1]
OUTPUT_H = f"data/processed/weather_data/{target}_hourly_{tag}.csv"

# ======================
# Load Data
# ======================

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": LATITUDE,
	"longitude": LONGITUDE,
	"start_date": START,
	"end_date": END,
	"daily": ["temperature_2m_mean", "rain_sum", "wind_speed_10m_max", "precipitation_sum"],
	"hourly": ["temperature_2m", "precipitation", "rain", "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m", "apparent_temperature", "cloud_cover"],
	"timezone": "America/Los_Angeles",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(4).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(5).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()

# Now we're gonna build in utc, then convert to LA, then drop tz
idx_utc = pd.date_range(
    start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
    end  =pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq =pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left",
)
idx_local = idx_utc.tz_convert("America/Los_Angeles")
idx = idx_local.tz_localize(None)  # naive local time

# Create DataFrame dictionary
hourly_data = {"datetime": idx}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["cloud_cover"] = hourly_cloud_cover

# # === LENGTH CHECK — HOURLY ===
_expected_h = len(idx)
_h_arrays = [
    hourly_temperature_2m, hourly_precipitation, hourly_rain,
    hourly_wind_speed_10m, hourly_wind_gusts_10m,
    hourly_relative_humidity_2m, hourly_apparent_temperature, hourly_cloud_cover
]
assert all(len(a) == _expected_h for a in _h_arrays), \
    f"[Hourly] Length mismatch: index={_expected_h}, vars={[len(a) for a in _h_arrays]}"

# Create dataframe
hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe["region"] = target

# Helper timestamps for DST handling
hourly_dataframe["ts_utc"] = idx_utc
hourly_dataframe["ts_local"] = idx   # same as 'datetime' but explicit


# ======================
# Hourly Reformatting
# ======================

# Standardize column names for modeling
hourly_dataframe = hourly_dataframe.rename(columns={
    "temperature_2m":       "temperature_c",
    "wind_speed_10m":       "wind_speed_ms",
    "wind_gusts_10m":       "wind_gust_ms",
    "relative_humidity_2m": "rel_humidity",
})

# Model-ready time keys
hourly_dataframe["date"]         = hourly_dataframe["ts_local"].dt.date
hourly_dataframe["hour"]         = hourly_dataframe["ts_local"].dt.hour
hourly_dataframe["month"]        = hourly_dataframe["ts_local"].dt.month
hourly_dataframe["weekday"]      = hourly_dataframe["ts_local"].dt.weekday
hourly_dataframe["weekend_flag"] = hourly_dataframe["weekday"].isin([5, 6]).astype(int)


# ==============================================
# Handle DST duplicate hour (Daylight's Savings.)
# ==============================================
hourly_before = len(hourly_dataframe)

# Sort by UTC so we always keep the earlier 01:00 when the clock falls back
hourly_dataframe = hourly_dataframe.sort_values(["region", "ts_utc"])

hourly_dataframe = hourly_dataframe.drop_duplicates(
    subset=["region", "date", "hour"], keep="first"
)

hourly_after = len(hourly_dataframe)
print(f"Removed {hourly_before - hourly_after} DST duplicate hour(s). "
      f"Final hourly rows: {hourly_after}")

# === QC FLAGS — HOURLY ===

hourly_dataframe["missing_weather_flag"] = hourly_dataframe[
    ["temperature_c", "precipitation", "wind_speed_ms"]
].isna().any(axis=1).astype(int)


print(hourly_dataframe.info())

# ======================
# Export to CSV
# ======================

cols_out = [
    "region","date","hour","month","weekday","weekend_flag",
    "temperature_c","apparent_temperature","rel_humidity",
    "wind_speed_ms","wind_gust_ms","cloud_cover",
    "precipitation","rain"
]
cols_out = [c for c in cols_out if c in hourly_dataframe.columns]
hourly_dataframe[cols_out].to_csv(
    OUTPUT_H,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)

print(f"Saved {len(hourly_dataframe)} hourly datapoints to {OUTPUT_H}.")
