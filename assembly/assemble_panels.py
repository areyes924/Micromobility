import pandas as pd
import numpy as np
from datetime import datetime

'''
Metro Bike Trip + Weather Panel Assembly Script

[Summary]

Notes:
----------------------
-

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

TRIP_DATAPATH = "data/processed/metro_trips/metro_trips_24-25.csv"
STATIONS_DATAPATH = "data/raw/metro-bike-share-stations.csv"
WEATHER_DATAPATH = "data/processed/weather_data"
REGIONS = ["Westside", "DTLA", "North Hollywood"]

HOURLY_OUTPUT_PATH = "data/processed/panels/hourly.csv"
DAILY_OUTPUT_PATH = "data/processed/panels/daily.csv"


# ======================
# Load Data
# ======================

hourly_weather = pd.concat(
    [pd.read_csv(f"{WEATHER_DATAPATH}/{r}_hourly_24-25.csv") for r in REGIONS],
    ignore_index=True
)

daily_weather = pd.concat(
    [pd.read_csv(f"{WEATHER_DATAPATH}/{r}_daily_24-25.csv") for r in REGIONS],
    ignore_index=True
)

trips = pd.read_csv(TRIP_DATAPATH)
stations = pd.read_csv(STATIONS_DATAPATH)

# ======================
# Adjust trips for easy merge
# ======================

trips["start_time"] = pd.to_datetime(trips["start_time"])
trips["end_time"] = pd.to_datetime(trips["end_time"])

# Extract time features
trips["date"] = trips["start_time"].dt.date
trips["hour"] = trips["start_time"].dt.hour
trips["weekday"] = trips["start_time"].dt.weekday  # Monday=0
trips["weekend_flag"] = trips["weekday"].isin([5, 6]).astype(int)
trips["month"] = trips["start_time"].dt.month



# Haversine formula for distance given lat/long
'''
hav(theta) = hav(change in latitude) + cos(lat1) * cos(lat2) * hav(hange in longitude)
hav(x) = (1 - cos(x)) / 2
theta = d/r
Solve for d

d = r * 2r arcsin (sqrt(hav(theta)))

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

print(f"There are {distance_hav(34.05661, 34.022449, -118.237213, -118.438332)} miles between Union Station and westside")  # accurate

trips["distance_miles"] = distance_hav(trips["start_lat"], trips["end_lat"], trips["start_lon"], trips["end_lon"])

avg_dist = trips["distance_miles"].mean()
print(f"Average trip distance is {avg_dist} miles.")

# Add "Region" to Trips

stations = stations.rename(columns={"Kiosk ID": "start_station"})

# print(stations.info())

# Merge region info into trips using start_station
trips = trips.merge(stations[["start_station", "Region"]], on="start_station", how="left")
trips.rename(columns={"Region": "region"}, inplace=True)
# Check for any unmatched stations
unmatched = trips[trips["region"].isna()]["start_station"].unique()
print("Unmatched start stations:", unmatched)

top_5_regions = trips['region'].value_counts().head(5)
print(top_5_regions)

# ======================
# Create Hourly Panel
# ======================

# Weather: make date a datetime.date, hour an int, tidy region text
hourly_weather["date"] = pd.to_datetime(hourly_weather["date"], errors="coerce").dt.date
hourly_weather["hour"] = pd.to_numeric(hourly_weather["hour"], errors="coerce").astype("Int64").astype(int)
hourly_weather["region"] = hourly_weather["region"].astype(str).str.strip()

# Trips/Ridership: ensure hour int, region tidy
trips["hour"] = trips["hour"].astype(int)
trips["region"] = trips["region"].astype(str).str.strip()

hourly_ridership = (
    trips.groupby(["region", "date", "hour"], as_index=False)
    .agg(
        trip_count = ("trip_id", "count"),
        avg_distance_mi = ("distance_miles", "mean"),
        avg_duration_min = ("duration", "mean"),
        weekend_flag = ("weekend_flag", "max"),
        month = ("month", "first"),
        weekday = ("weekday", "first")
    )
)

# Merge with weather

hourly_merged = pd.merge(
    hourly_ridership,
    hourly_weather,
    left_on=["region", "date", "hour"],
    right_on=["region", "date", "hour"],
    how="left"
)

hourly_merged["missing_weather_flag"] = hourly_merged["temperature_c"].isna().astype(int)

# ======================
# Clean Hourly Panel
# ======================

hourly_merged.rename(columns={"weekend_flag_x": "weekend_flag", "month_x":"month", "weekday_x":"weekday"}, inplace=True)
hourly_merged.drop(["month_y", "weekday_y", "weekend_flag_y"], axis=1, inplace=True)

# ======================
# Create Daily Panel
# ======================

daily_panel = (
    hourly_merged.groupby(["region", "date"], as_index=False)
    .agg(
        trip_count=("trip_count", "sum"),
        avg_distance_mi=("avg_distance_mi", "mean"),
        avg_duration_min=("avg_duration_min", "mean"),
        temperature_c_mean=("temperature_c", "mean"),
        rain_sum=("precipitation", "sum"),
        wind_speed_ms_max=("wind_speed_ms", "max"),
        month=("month", "first"),
        weekday=("weekday", "first"),
        weekend_flag=("weekend_flag", "max")
    )
)

# ======================
# Export as CSV
# ======================
"""
hourly_merged.to_csv(
    HOURLY_OUTPUT_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)
print(f"Exported {len(hourly_merged)} datapoints to {HOURLY_OUTPUT_PATH}.")

daily_panel.to_csv(
    DAILY_OUTPUT_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)

print(f"Exported {len(daily_panel)} datapoints to {DAILY_OUTPUT_PATH}.")
"""
# ======================
# Visualizations
# ======================

import matplotlib.pyplot as plt

# Trips by hour (region breakdown)
hourly_merged.groupby(["hour", "region"])["trip_count"].mean().unstack().plot()
plt.title("Average Hourly Trips by Region")
plt.ylabel("Trips per Hour")
plt.savefig("plots/hourly_trips_by_region.png")

# Temperature vs Trips
plt.scatter(hourly_merged["temperature_c"], hourly_merged["trip_count"], alpha=0.3)
plt.title("Temperature vs Hourly Trips")
plt.xlabel("Temperature (°C)")
plt.ylabel("Trip Count")
plt.savefig("plots/temp_vs_trips_scatter.png")
