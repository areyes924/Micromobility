import pandas as pd
import numpy as np

'''
UCLA Metro Bike Trip Data Cleaning Script

This script consolidates, cleans, and filters multiple quarterly Metro Bike
trip datasets for the 2024–2025 academic year into unified, analysis-ready CSVs.
, and also isolates trips related to UCLA stations.

Notes:
----------------------
- All input CSVs have identical headers and consistent schemas.
- All numeric columns were prevalidated and are numbers.
- All timestamps are in local time.

===========================================================
'''

# ======================
# Define Constant Parameters
# ======================

UCLA_OUTPUT_PATH = "data/processed/metro_trips_ucla_24-25.csv"
GENERAL_OUTPUT_PATH = "data/processed/metro_trips_24-25.csv"
ACADEMIC_START = pd.Timestamp("2024-09-23")
ACADEMIC_END   = pd.Timestamp("2025-09-22")
VIRTUAL_STATION = 3000
UCLA_IDS = [4614, 4643, 4613]  # perloff, drake, gateway

ROUND_TRIP_BOUNDARY = 15 # Delete all round trips under this number of minutes
MAX_HOURS = 10 # Maximum hours 

# ======================
# Load + Merge Unclean Data
# ======================

# Note that these all have identical column headers, making everything much easier.
q3_2024 = pd.read_csv("data/raw/metro-trips-2024-q3.csv")
q4_2024 = pd.read_csv("data/raw/metro-trips-2024-q4.csv")
q1_2025 = pd.read_csv("data/raw/metro-trips-2025-q1.csv")
q2_2025 = pd.read_csv("data/raw/metro-trips-2025-q2.csv")
q3_2025 = pd.read_csv("data/raw/metro-trips-2025-q3.csv")
trips = pd.concat((q3_2024, q4_2024, q1_2025, q2_2025, q3_2025), ignore_index=True)

# ======================
# Delete Unnecessary Columns and Standardize Names
# ======================

# Drop

trips = trips.drop(['plan_duration', 'trip_route_category'], axis=1,errors='ignore')

# Rearrange
trips = trips[['trip_id', 'start_station', 'end_station', 'start_time', 'end_time', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 'duration', 'passholder_type','bike_id', 'bike_type']]

# ======================
# Strip Unnecessary Datapoints 
# ======================

# Drop Na values
initial_count = len(trips)
trips = trips.dropna(subset=['start_time','end_time','start_station','end_station','duration'])
print(f"Dropped {initial_count - len(trips)} null values.")

# Delete Virtual Station Entries

trips = trips[
    (trips['start_station'] != VIRTUAL_STATION) &
    (trips['end_station'] != VIRTUAL_STATION)
].copy()

# Delete Duplicates

initial_count = len(trips)
trips = trips.drop_duplicates(
    subset=[
        'start_time', 'end_time',
        'start_station', 'end_station',
        'bike_id'
    ]
)
print(f"Dropped {initial_count - len(trips)} duplicates.")

# Delete Trips with Unrealistic Duration Numbers
initial_count = len(trips)
trips = trips[trips['duration'] <= (MAX_HOURS * 60)].copy()
print(f"Dropped {initial_count - len(trips)} unrealistically long trips.")

# Delete Trips with the same start/end station (someone took out a bike and just put it back)
initial_count = len(trips)
trips = trips[
    (trips["start_station"] != trips["end_station"]) |
    (trips["duration"] >= ROUND_TRIP_BOUNDARY)
].copy()
print(f"Dropped {initial_count - len(trips)} accidental rides.")

# Remove trips outside desired academic calendar
initial_count = len(trips)
trips["start_time"] = pd.to_datetime(trips["start_time"])
trips["end_time"] = pd.to_datetime(trips["end_time"])
mask = (trips['start_time'] >= ACADEMIC_START) & (trips['start_time'] < ACADEMIC_END)
trips = trips.loc[mask].copy()
print(f"Dropped {initial_count - len(trips)} rides outside of the academic year, {initial_count} were there before")

# ======================
# Isolate UCLA Trips
# ======================

ucla_trips = trips[
    (trips["start_station"].isin(UCLA_IDS)) |
    (trips["end_station"].isin(UCLA_IDS))
].copy()
ucla_trips["ucla_trip_type"] = np.select(
    [
        (ucla_trips["start_station"].isin(UCLA_IDS)) & (~ucla_trips["end_station"].isin(UCLA_IDS)),
        (ucla_trips["end_station"].isin(UCLA_IDS)) & (~ucla_trips["start_station"].isin(UCLA_IDS)),
    ],
    ["From UCLA", "To UCLA"],
    default="UCLA to UCLA"
)
print(f"Extracted {len(ucla_trips)} trips involving UCLA stations "
      f"({len(ucla_trips)/len(trips):.2%} of total).")

# ======================
# Export
# ======================

ucla_trips.to_csv(
    UCLA_OUTPUT_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)

print(f"Saved {len(ucla_trips)} UCLA specific rides to {UCLA_OUTPUT_PATH}.")

trips.to_csv(
    GENERAL_OUTPUT_PATH,
    index=False,         # don't write the row index
    encoding="utf-8",    # standard text encoding
    float_format="%.6f"
)

print(f"Saved {len(trips)} rides to {GENERAL_OUTPUT_PATH}.")
