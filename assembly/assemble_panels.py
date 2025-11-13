import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import assembly.constants as constants
'''
Metro Bike Trip + Weather Panel Assembly Script

This script merges Metro trip records with station regions, computes trip
distances (Haversine), adds time features, and aggregates ridership hourly.
It joins hourly ridership to region-specific Open-Meteo weather, builds a
clean hourly panel, and exports a standardized CSV for modeling.

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

tag = constants.TAG

TRIP_DATAPATH = f"data/processed/metro_trips/metro_trips_{tag}.csv"
STATIONS_DATAPATH = "data/raw/metro-bike-share-stations.csv"
WEATHER_DATAPATH = "data/processed/weather_data"
REGIONS = ["Westside", "DTLA", "North Hollywood"]

HOURLY_OUTPUT_PATH = f"data/processed/panels/hourly_{tag}.csv"

# ======================
# Load Data
# ======================

hourly_weather = pd.concat(
    [pd.read_csv(f"{WEATHER_DATAPATH}/{r}_hourly_{tag}.csv") for r in REGIONS],
    ignore_index=True
)

trips = pd.read_csv(TRIP_DATAPATH)
stations = pd.read_csv(STATIONS_DATAPATH)

# ======================
# Canonical Typing and Normalization
# ======================

# Region should be string
hourly_weather["region"] = hourly_weather["region"].astype(str).str.strip()

# Hourly weather date and hour
hourly_weather["date"] = pd.to_datetime(hourly_weather["date"], errors="coerce").dt.date
hourly_weather["hour"] = pd.to_numeric(hourly_weather["hour"], errors="coerce").astype(int)

# Clarify that precipitation and rain are both in millimeters
if "precipitation" in hourly_weather.columns:
    hourly_weather = hourly_weather.rename(columns={"precipitation": "precip_mm"})
if "rain" in hourly_weather.columns:
    hourly_weather = hourly_weather.rename(columns={"rain": "rain_mm"})

# Ensure that weather variables are numeric
for c in ["precip_mm", "rain_mm", "temperature_c", "wind_speed_ms", "wind_gust_ms",
          "rel_humidity", "apparent_temperature", "cloud_cover"]:
    if c in hourly_weather.columns:
        hourly_weather[c] = pd.to_numeric(hourly_weather[c], errors="coerce")

# ======================
# Prepare Trips
# ======================

trips["start_time"] = pd.to_datetime(trips["start_time"], errors="coerce")
trips["end_time"] = pd.to_datetime(trips["end_time"], errors="coerce")

# Extract time features
trips["date"] = trips["start_time"].dt.date
trips["hour"] = trips["start_time"].dt.hour.astype(int)
trips["weekday"] = trips["start_time"].dt.weekday.astype(int)  # Monday = 0
trips["weekend_flag"] = trips["weekday"].isin([5, 6]).astype(int)
trips["month"] = trips["start_time"].dt.month.astype(int)

# Haversine formula for distance given lat/long
'''
hav(theta) = hav(change in latitude) + cos(lat1) * cos(lat2) * hav(change in longitude)
hav(x) = (1 - cos(x)) / 2
theta = d/r
Solve for d

d = 2r arcsin (sqrt(hav(theta)))

'''

def haversine(x):
    return (1 - np.cos(x)) / 2

def distance_hav(lat1, lat2, long1, long2):
    # radian conversions
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    long1 = np.radians(long1)
    long2 = np.radians(long2)

    delta_lat = (lat2-lat1)
    delta_long = (long2-long1)
    rad = 3958.8

    hav_theta = haversine(delta_lat) + np.cos(lat1) * np.cos(lat2) * haversine(delta_long)
    return 2 * rad * np.arcsin(np.sqrt(hav_theta))

# print(f"There are {distance_hav(34.05661, 34.022449, -118.237213, -118.438332)} miles between Union Station and westside")  # accurate

trips["distance_miles"] = distance_hav(trips["start_lat"], trips["end_lat"], trips["start_lon"], trips["end_lon"])
avg_dist = trips["distance_miles"].mean()
print(f"Average trip distance is {avg_dist} miles.")

# ======================
# Add Region to Trips
# ======================

stations = stations.rename(columns={"Kiosk ID": "start_station"})
trips = trips.merge(stations[["start_station", "Region"]], on="start_station", how="left")
trips = trips.rename(columns={"Region": "region"})
trips["region"] = trips["region"].astype(str).str.strip()

# Keep only regions present in weather to avoid orphaned rows
trips = trips[trips["region"].isin(REGIONS)]

# ======================
# Create Hourly Panel
# ======================

# Aggregate trips at region-date-hour
hourly_ridership = (
    trips.groupby(["region", "date", "hour"], as_index=False)
         .agg(
             trip_count=("trip_id", "count"),
             avg_distance_mi=("distance_miles", "mean"),
             avg_duration_min=("duration", "mean"),
         )
)

# Merge weather with trips so zero trip hours are kept now
hourly_merged = hourly_weather.merge(
    hourly_ridership,
    on=["region", "date", "hour"],
    how="left"
)
if "trip_count" in hourly_merged.columns:
    hourly_merged["trip_count"] = hourly_merged["trip_count"].fillna(0).astype(int)

# Keep averages as NaN when no trips
for c in ["avg_distance_mi", "avg_duration_min"]:
    if c in hourly_merged.columns:
        # leave NaN, but ensure float dtype
        hourly_merged[c] = pd.to_numeric(hourly_merged[c], errors="coerce")

# Expect full grid: hours_per_region * number_of_regions
print("Rows after merge:", len(hourly_merged))
print("Unique keys:", hourly_merged[["region","date","hour"]].drop_duplicates().shape[0])

# Sanity: no duplicate keys
assert not hourly_merged.duplicated(["region","date","hour"]).any()

# Mark rows with missing weather
hourly_merged["missing_weather_flag"] = hourly_merged["temperature_c"].isna().astype(int)

# ======================
# Clean Hourly Panel
# ======================

# Normalize date to ISO string once
hourly_merged["date"] = pd.to_datetime(hourly_merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")

# Build stable zero-based date_id by sorting unique dates
unique_dates = sorted(hourly_merged["date"].dropna().unique())
date_to_id = {d: i for i, d in enumerate(unique_dates)}
hourly_merged["date_id"] = hourly_merged["date"].map(date_to_id).astype(int)

# Order columns nicely
hourly_cols = [
    "region", "date", "date_id", "hour",
    "month", "weekday", "weekend_flag",
    "trip_count", "avg_distance_mi", "avg_duration_min",
    "temperature_c", "apparent_temperature", "rel_humidity",
    "wind_speed_ms", "wind_gust_ms", "cloud_cover",
    "precip_mm", "rain_mm",
    "missing_weather_flag",
]
hourly_cols = [c for c in hourly_cols if c in hourly_merged.columns]
hourly_merged = hourly_merged[hourly_cols]

# ======================
# Final Cleanup and Export
# ======================

# Round float columns for consistent CSVs
float_cols = [
    c for c in [
        "avg_distance_mi", "avg_duration_min",
        "temperature_c", "apparent_temperature", "rel_humidity",
        "wind_speed_ms", "wind_gust_ms", "cloud_cover",
        "precip_mm", "rain_mm"
    ] if c in hourly_merged.columns
]
hourly_merged[float_cols] = hourly_merged[float_cols].round(3)

# Export hourly panel
hourly_merged.to_csv(HOURLY_OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Exported {len(hourly_merged)} datapoints to {HOURLY_OUTPUT_PATH}.")

# ======================
# Visualizations
# ======================

# Trips by hour (region breakdown)
hourly_merged.groupby(["hour", "region"])["trip_count"].mean().unstack().plot()
plt.title("Average Hourly Trips by Region")
plt.ylabel("Trips per Hour")
plt.savefig("plots/hourly_trips_by_region.png")

# Temperature vs Trips
plt.scatter(hourly_merged["temperature_c"], hourly_merged["trip_count"], alpha=0.3,color = "0.3")
plt.title("Temperature vs Hourly Trips")
plt.xlabel("Temperature (°C)")
plt.ylabel("Trip Count")
plt.savefig("plots/temp_vs_trips_scatter.png")


