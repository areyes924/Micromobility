import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants

'''
NYC Citi Bike + Weather Panel Assembly Script (streaming)

- Reads hourly Open-Meteo weather by region.
- Streams raw Citi Bike CSVs from data/raw/NYC/ in chunks.
- Assigns "region" via NYC bounding box + rectangular boxes:
  NYC_South, NYC_Middle, NYC_East, NYC_North, else Other.
- Drops round trips under 15 minutes (start_station_id == end_station_id).
- Aggregates to region x date x hour (trip_count, avg_duration_min).
- Merges onto full weather grid to preserve zero-trip hours.
- Adds date_id and missing_weather_flag.
- Exports standardized hourly panel for modeling.

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

tag = constants.TAG
START = pd.Timestamp(constants.START)
END = pd.Timestamp(constants.END)

RAW_TRIP_DIR = "data/raw/NYC"
WEATHER_DIR = "data/processed/weather_data"
HOURLY_OUTPUT_PATH = f"data/processed/panels/NYC_hourly_{tag}.csv"

REGIONS = ["NYC_South", "NYC_Middle", "NYC_East", "NYC_North"]

# Cleaning thresholds
MAX_HOURS = 10
MIN_SECONDS = 60
ROUND_TRIP_BOUNDARY_MIN = 15

# Streaming
CHUNKSIZE = 250_000

# NYC bounding box
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

# ======================
# Region boxes (partition))
# ======================
# Priority order matters. East should usually be checked first so it captures Queens-ish points.

# East: carve out an "east slice" (Queens / LIC / etc)
EAST_LON_MIN = -73.95  # anything east of this is "NYC_East" (tune if needed)

# South/Middle/North latitude bands for the non-east remainder
SOUTH_LAT_MAX = 40.72
MIDDLE_LAT_MAX = 40.80
# North is >= MIDDLE_LAT_MAX

def assign_region_boxes(start_lat, start_lon):
    lat = pd.to_numeric(start_lat, errors="coerce")
    lon = pd.to_numeric(start_lon, errors="coerce")

    out = np.array(["Other"] * len(lat), dtype=object)

    in_bounds = (
        lat.between(NYC_LAT_MIN, NYC_LAT_MAX) &
        lon.between(NYC_LON_MIN, NYC_LON_MAX)
    )

    if in_bounds.sum() == 0:
        return out

    latv = lat.to_numpy()
    lonv = lon.to_numpy()
    inb = in_bounds.to_numpy()

    # East first
    east = inb & (lonv >= EAST_LON_MIN)
    out[east] = "NYC_East"

    # Remaining (non-east) split by latitude into south/middle/north
    rem = inb & (~east)

    south = rem & (latv < SOUTH_LAT_MAX)
    middle = rem & (latv >= SOUTH_LAT_MAX) & (latv < MIDDLE_LAT_MAX)
    north = rem & (latv >= MIDDLE_LAT_MAX)

    out[south] = "NYC_South"
    out[middle] = "NYC_Middle"
    out[north] = "NYC_North"

    return out

# ======================
# Load Weather
# ======================

hourly_weather = pd.concat(
    [pd.read_csv(f"{WEATHER_DIR}/{r}_hourly_{tag}.csv") for r in REGIONS],
    ignore_index=True
)

# ======================
# Weather normalization
# ======================

hourly_weather["region"] = hourly_weather["region"].astype(str).str.strip()
hourly_weather["date"] = pd.to_datetime(hourly_weather["date"], errors="coerce").dt.date
hourly_weather["hour"] = pd.to_numeric(hourly_weather["hour"], errors="coerce").astype(int)

# Standardize precip/rain names to *_mm
if "precipitation" in hourly_weather.columns:
    hourly_weather = hourly_weather.rename(columns={"precipitation": "precip_mm"})
if "rain" in hourly_weather.columns:
    hourly_weather = hourly_weather.rename(columns={"rain": "rain_mm"})

# Numeric casting for core weather vars (if present)
for c in [
    "temperature_c", "apparent_temperature", "rel_humidity",
    "wind_speed_ms", "wind_gust_ms", "cloud_cover",
    "precip_mm", "rain_mm"
]:
    if c in hourly_weather.columns:
        hourly_weather[c] = pd.to_numeric(hourly_weather[c], errors="coerce")

# Snow handling
# If your weather grabber adds hourly "snowfall" (cm), convert to water-equivalent mm:
# Open-Meteo guidance: 7 cm snow ~ 10 mm water => mm = cm * (10/7)
if "snowfall" in hourly_weather.columns:
    hourly_weather["snow_mm"] = pd.to_numeric(hourly_weather["snowfall"], errors="coerce") * (10.0 / 7.0)
else:
    # Fallback if you do not have snowfall:
    # snow_mm = max(precip_mm - rain_mm, 0)
    if "precip_mm" in hourly_weather.columns and "rain_mm" in hourly_weather.columns:
        hourly_weather["snow_mm"] = (hourly_weather["precip_mm"] - hourly_weather["rain_mm"]).clip(lower=0)
    else:
        hourly_weather["snow_mm"] = 0.0

hourly_weather["snow_mm"] = pd.to_numeric(hourly_weather["snow_mm"], errors="coerce").fillna(0.0)

# Fill rain/precip missing to 0 for convenience (optional)
if "rain_mm" in hourly_weather.columns:
    hourly_weather["rain_mm"] = hourly_weather["rain_mm"].fillna(0.0)
if "precip_mm" in hourly_weather.columns:
    hourly_weather["precip_mm"] = hourly_weather["precip_mm"].fillna(0.0)

# Keep only expected regions
hourly_weather = hourly_weather[hourly_weather["region"].isin(REGIONS)].copy()

# If time features are missing, derive them from date
need_time_features = any(col not in hourly_weather.columns for col in ["month", "weekday", "weekend_flag"])
if need_time_features:
    dt = pd.to_datetime(hourly_weather["date"], errors="coerce")
    hourly_weather["month"] = dt.dt.month.astype(int)
    hourly_weather["weekday"] = dt.dt.weekday.astype(int)
    hourly_weather["weekend_flag"] = hourly_weather["weekday"].isin([5, 6]).astype(int)

# ======================
# Stream raw trips and accumulate hourly totals
# ======================

files = sorted([f for f in os.listdir(RAW_TRIP_DIR) if f.endswith(".csv")])
print(f"Found {len(files)} raw NYC trip files in {RAW_TRIP_DIR}")
if len(files) == 0:
    raise FileNotFoundError(f"No CSV files found in {RAW_TRIP_DIR}")

usecols = [
    "ride_id", "rideable_type",
    "started_at", "ended_at",
    "start_station_id", "end_station_id",
    "start_lat", "start_lng"
]

# Accumulator: (region, date, hour) -> [trip_count, sum_duration_min]
acc = {}

rows_seen = 0
rows_used = 0
rows_dropped_roundtrip = 0

for fname in files:
    fp = f"{RAW_TRIP_DIR}/{fname}"
    print("Reading:", fname)

    reader = pd.read_csv(fp, usecols=lambda c: c in usecols, chunksize=CHUNKSIZE, low_memory=False)

    for chunk in reader:
        rows_seen += len(chunk)

        # Standardize names
        chunk = chunk.rename(columns={
            "started_at": "start_time",
            "ended_at": "end_time",
            "start_lng": "start_lon"
        })

        # Drop core nulls
        chunk = chunk.dropna(subset=["ride_id", "start_time", "end_time", "start_lat", "start_lon"])
        if len(chunk) == 0:
            continue

        # Parse timestamps
        chunk["start_time"] = pd.to_datetime(chunk["start_time"], errors="coerce")
        chunk["end_time"] = pd.to_datetime(chunk["end_time"], errors="coerce")
        chunk = chunk.dropna(subset=["start_time", "end_time"])
        if len(chunk) == 0:
            continue

        # Filter by window early
        chunk = chunk[(chunk["start_time"] >= START) & (chunk["start_time"] < END)].copy()
        if len(chunk) == 0:
            continue

        # Duration cleaning
        dur_sec = (chunk["end_time"] - chunk["start_time"]).dt.total_seconds()
        chunk = chunk[(dur_sec >= MIN_SECONDS) & (dur_sec <= MAX_HOURS * 3600)].copy()
        if len(chunk) == 0:
            continue
        chunk["duration_min"] = dur_sec / 60.0

        # Drop short round trips: same station id + short duration
        if "start_station_id" in chunk.columns and "end_station_id" in chunk.columns:
            same_station = chunk["start_station_id"].astype(str) == chunk["end_station_id"].astype(str)
            short_round = same_station & (chunk["duration_min"] < ROUND_TRIP_BOUNDARY_MIN)
            dropped = int(short_round.sum())
            if dropped > 0:
                rows_dropped_roundtrip += dropped
                chunk = chunk[~short_round].copy()
            if len(chunk) == 0:
                continue

        # Time keys
        chunk["date"] = chunk["start_time"].dt.date
        chunk["hour"] = chunk["start_time"].dt.hour.astype(int)

        # Region assignment from start coords
        chunk["region"] = assign_region_boxes(chunk["start_lat"], chunk["start_lon"])
        chunk = chunk[chunk["region"].isin(REGIONS)].copy()
        if len(chunk) == 0:
            continue

        rows_used += len(chunk)

        # Aggregate within chunk: count + sum duration
        g = (
            chunk.groupby(["region", "date", "hour"], as_index=False)
                 .agg(
                     trip_count=("ride_id", "count"),
                     sum_duration_min=("duration_min", "sum")
                 )
        )

        # Accumulate into global dict
        for r in g.itertuples(index=False):
            key = (r.region, r.date, int(r.hour))
            if key not in acc:
                acc[key] = [int(r.trip_count), float(r.sum_duration_min)]
            else:
                acc[key][0] += int(r.trip_count)
                acc[key][1] += float(r.sum_duration_min)

print(f"Rows seen: {rows_seen:,}")
print(f"Rows used after filters: {rows_used:,}")
print(f"Rows dropped (short round trips): {rows_dropped_roundtrip:,}")
print(f"Unique region-date-hour keys accumulated: {len(acc):,}")

# ======================
# Build Hourly Ridership DataFrame
# ======================

ridership_rows = []
for (region, date, hour), (cnt, sum_dur) in acc.items():
    avg_dur = (sum_dur / cnt) if cnt > 0 else np.nan
    ridership_rows.append([region, date, hour, cnt, avg_dur])

hourly_ridership = pd.DataFrame(
    ridership_rows,
    columns=["region", "date", "hour", "trip_count", "avg_duration_min"]
)

# ======================
# Merge Weather with Ridership (keep weather grid)
# ======================

hourly_merged = hourly_weather.merge(
    hourly_ridership,
    on=["region", "date", "hour"],
    how="left"
)

hourly_merged["trip_count"] = hourly_merged["trip_count"].fillna(0).astype(int)
hourly_merged["avg_duration_min"] = pd.to_numeric(hourly_merged["avg_duration_min"], errors="coerce")

print("Rows after merge:", len(hourly_merged))
print("Unique keys:", hourly_merged[["region", "date", "hour"]].drop_duplicates().shape[0])
assert not hourly_merged.duplicated(["region", "date", "hour"]).any()

# Missing weather flag (match your existing convention)
hourly_merged["missing_weather_flag"] = hourly_merged["temperature_c"].isna().astype(int)

# ======================
# date_id
# ======================

hourly_merged["date"] = pd.to_datetime(hourly_merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
unique_dates = sorted(hourly_merged["date"].dropna().unique())
date_to_id = {d: i for i, d in enumerate(unique_dates)}
hourly_merged["date_id"] = hourly_merged["date"].map(date_to_id).astype(int)

# ======================
# Order columns to match modeling expectations
# ======================

hourly_cols = [
    "region", "date", "date_id", "hour",
    "month", "weekday", "weekend_flag",
    "trip_count", "avg_duration_min",
    "temperature_c", "apparent_temperature", "rel_humidity",
    "wind_speed_ms", "wind_gust_ms", "cloud_cover",
    "precip_mm", "rain_mm", "snow_mm",
    "missing_weather_flag",
]
hourly_cols = [c for c in hourly_cols if c in hourly_merged.columns]
hourly_merged = hourly_merged[hourly_cols]

# Round floats for consistent CSVs
float_cols = [
    c for c in [
        "avg_duration_min",
        "temperature_c", "apparent_temperature", "rel_humidity",
        "wind_speed_ms", "wind_gust_ms", "cloud_cover",
        "precip_mm", "rain_mm", "snow_mm"
    ] if c in hourly_merged.columns
]
hourly_merged[float_cols] = hourly_merged[float_cols].round(3)

# ======================
# Export
# ======================

os.makedirs("data/processed/panels", exist_ok=True)
hourly_merged.to_csv(HOURLY_OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Exported {len(hourly_merged)} datapoints to {HOURLY_OUTPUT_PATH}.")

# ======================
# Visualizations
# ======================

os.makedirs("plots/assembly", exist_ok=True)

hourly_merged.groupby(["hour", "region"])["trip_count"].mean().unstack().plot()
plt.title("Average Hourly Trips by Region (NYC)")
plt.ylabel("Trips per Hour")
plt.savefig("plots/assembly/NYC_hourly_trips_by_region.png")
plt.close()

plt.scatter(hourly_merged["temperature_c"], hourly_merged["trip_count"], alpha=0.3, color="0.3")
plt.title("Temperature vs Hourly Trips (NYC)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Trip Count")
plt.savefig("plots/assembly/NYC_temp_vs_trips_scatter.png")
plt.close()
