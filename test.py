import pandas as pd
stations = pd.read_csv("data/raw/metro-bike-share-stations.csv")
trips = pd.read_csv("data/processed/metro_trips_ucla_24-25.csv")

print("===== Stations dtypes =====")
print(pd.DataFrame(stations.dtypes, columns=['dtype']))
print("===== Trips dtypes =====")
print(pd.DataFrame(trips.dtypes, columns=['dtype']))