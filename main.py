import pandas as pd

# ==================================================
# Data Loading
# ==================================================

station_map = pd.read_csv("data/raw/metro-bike-share-stations-2025-10-01.csv")
q3_2025 = pd.read_csv("data/raw/metro-trips-2025-q3.csv")
q2_2025 = pd.read_csv("data/raw/metro-trips-2025-q2.csv")

df = q2_2025 # When testing, it's nice to just focus on one.

# Store relevant station IDs. We could do this in pandas using station_map

perloff_id = 4614 # North Campus Station near Bunche/Perloff Halls
drake_id = 4643 # Drake Stadium Station near Epicuria/Bfit
gateway_id1 = 4613 # Gateway Plaza Station near Bus stops/Pauley Pavilion
gateway_id2 = 4665 # For some reason, there are two station IDs for gateway plaza. This second one doesn't seem to be in use.

# ==================================================
# Cleaning
# ==================================================


# Contingency in case Gateway_id2 is used.
gateway_id2_rides = df[(df["start_station"] == gateway_id2) | (df["end_station"] == gateway_id2)]
if len(gateway_id2_rides) != 0:
    print(len(gateway_id2_rides) + " RIDES FROM THE OTHER ID OF GATEWAY PLAZA DETECTED. TAKE THIS INTO ACCOUNT")
else: print("Gateway ID2 check OK")

# print (station_map.info())
# print (q3_2025.info())

# ==================================================
# Analysis
# ==================================================

# Filter for rides that start at drake_id and end at gateway_id
rides = q2_2025[(q2_2025["start_station"] == drake_id) & (q2_2025["end_station"] == gateway_id1)]

# Count the total number of such rides
total_rides = len(rides)
print(f"Total rides from Drake to Bus Stop: {total_rides}")