import pandas as pd

# ==================================================
# Load Data
# ==================================================
stations = pd.read_csv("data/raw/metro-bike-share-stations.csv")
ucla_trips = pd.read_csv("data/processed/metro_trips_ucla_24-25.csv")
trips = pd.read_csv("data/processed/metro_trips_24-25.csv")

# Normalize headers
stations = stations.rename(columns={
    "Kiosk ID": "station_id",
    "Kiosk Name": "station_name"
})[["station_id", "station_name"]]

# ==================================================
# Map Station Names to Trips
# ==================================================
id_to_name = stations.set_index("station_id")["station_name"].to_dict()

# Create readable start/end station columns
ucla_trips["start_name"] = ucla_trips["start_station"].map(id_to_name)
ucla_trips["end_name"] = ucla_trips["end_station"].map(id_to_name)

# Drop trips missing station names
ucla_trips = ucla_trips.dropna(subset=["start_name", "end_name"])

trips["start_name"] = trips["start_station"].map(id_to_name)
trips["end_name"] = trips["end_station"].map(id_to_name)

# Drop trips missing station names
trips = trips.dropna(subset=["start_name", "end_name"])

# ==================================================
# Find Most Common Trips Involving UCLA
# ==================================================
top_trips = (
    ucla_trips.groupby(["start_name", "end_name"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# Display top 10 overall
print("\n=== Top 10 Most Common Trips Involving UCLA ===")
print(top_trips.head(10).to_string(index=False))

# ==================================================
# Breakdown by Direction (From vs To UCLA)
# ==================================================
if "ucla_trip_type" in ucla_trips.columns:
    print("\n=== Top 5 'From UCLA' Trips ===")
    from_ucla = (
        ucla_trips[ucla_trips["ucla_trip_type"] == "From UCLA"]
        .groupby(["start_name", "end_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
    )
    print(from_ucla.to_string(index=False))

    print("\n=== Top 5 'To UCLA' Trips ===")
    to_ucla = (
        ucla_trips[ucla_trips["ucla_trip_type"] == "To UCLA"]
        .groupby(["start_name", "end_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
    )
    print(to_ucla.to_string(index=False))

    print("\n=== Top 'UCLA to UCLA' Trips ===")
    ucla_to_ucla = (
        ucla_trips[ucla_trips["ucla_trip_type"] == "UCLA to UCLA"]
        .groupby(["start_name", "end_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(ucla_to_ucla.to_string(index=False))

print("\n=== Top General Trips ===")
general = (
    trips
    .groupby(["start_name", "end_name"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)
print(general.to_string(index=False))
