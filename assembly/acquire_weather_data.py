import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

'''
Los Angeles Weather Data Acquisition Script

[Summary]

https://open-meteo.com/en/docs/historical-weather-api?start_date=2024-09-23&end_date=2025-09-22&timezone=America%2FLos_Angeles&hourly=temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,relative_humidity_2m,apparent_temperature,cloud_cover&latitude=34.056278&longitude=-118.231773&daily=temperature_2m_mean,rain_sum,wind_speed_10m_max

Notes:
----------------------
- 
pip install openmeteo-requests
pip install requests-cache retry-requests numpy pandas

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

target = "Westside"

# These are all in the approximate center of the defined regions.
LAT_LONG = {
    "DTLA": (34.05661, -118.237213), # Union Station Coordinates
    "Westside": (34.022449,-118.438332), # Barrington and National Coordinates
    "NoHo": (34.16552,-118.375153) # North Hollywood Plaza Coordinates
}

LATITUDE = LAT_LONG[target][0]
LONGITUDE = LAT_LONG[target][1]
START = "2024-09-23"
END = "2025-09-22"
OUTPUT_H = f"data/processed/weather_data/{target}_hourly_24-25.csv"
OUTPUT_D = f"data/processed/weather_data/{target}_daily_24-25.csv"

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

idx = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True)
             .tz_convert("America/Los_Angeles"),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
             .tz_convert("America/Los_Angeles"),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
).tz_localize(None)  # Drop timezone to match trip dataset

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

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy()

# Build local-time daily index for Los Angeles
daily_idx = pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True)
             .tz_convert("America/Los_Angeles"),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
             .tz_convert("America/Los_Angeles"),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
).tz_localize(None)

daily_data = {"date": daily_idx.date}

daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
daily_data["rain_sum"] = daily_rain_sum
daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
daily_data["precipitation_sum"] = daily_precipitation_sum

# === LENGTH CHECK — DAILY ===
_expected_d = len(daily_idx)
_d_arrays = [
    daily_temperature_2m_mean, daily_rain_sum,
    daily_wind_speed_10m_max, daily_precipitation_sum
]
assert all(len(a) == _expected_d for a in _d_arrays), \
    f"[Daily] Length mismatch: index={_expected_d}, vars={[len(a) for a in _d_arrays]}"

# Create dataframe
daily_dataframe = pd.DataFrame(data = daily_data)
daily_dataframe["region"]  = target


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
hourly_dataframe["date"]         = hourly_dataframe["datetime"].dt.date
hourly_dataframe["hour"]         = hourly_dataframe["datetime"].dt.hour
hourly_dataframe["month"]        = hourly_dataframe["datetime"].dt.month
hourly_dataframe["weekday"]      = hourly_dataframe["datetime"].dt.weekday
hourly_dataframe["weekend_flag"] = hourly_dataframe["weekday"].isin([5, 6]).astype(int)

daily_dataframe = daily_dataframe.rename(columns={
    "temperature_2m_mean": "temperature_c_mean",
    "wind_speed_10m_max":  "wind_speed_ms_max",
    # keep *_sum names as-is for clarity
})

# ======================
# Daily Reformatting
# ======================

# Model-ready time keys (convert to datetime first)
daily_dataframe["date"]         = pd.to_datetime(daily_dataframe["date"])
daily_dataframe["month"]        = daily_dataframe["date"].dt.month
daily_dataframe["weekday"]      = daily_dataframe["date"].dt.weekday
daily_dataframe["weekend_flag"] = daily_dataframe["weekday"].isin([5, 6]).astype(int)
# === QC FLAGS — HOURLY ===

hourly_dataframe["missing_weather_flag"] = hourly_dataframe[
    ["temperature_c", "precipitation", "wind_speed_ms"]
].isna().any(axis=1).astype(int)


print(hourly_dataframe.info())



# === QC FLAGS — DAILY ===

daily_dataframe["missing_weather_flag"] = daily_dataframe[
    ["temperature_c_mean", "precipitation_sum"]
].isna().any(axis=1).astype(int)
print(daily_dataframe.info())


# ======================
# Export to CSV
# ======================

hourly_dataframe.to_csv(
    OUTPUT_H,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)

print(f"Saved {len(hourly_dataframe)} hourly datapoints to {OUTPUT_H}.")

daily_dataframe.to_csv(
    OUTPUT_D,
    index=False,         # don't write the row index
    encoding="utf-8",    # standard text encoding
    float_format="%.6f"
)

print(f"Saved {len(daily_dataframe)} daily datapoints to {OUTPUT_D}.")