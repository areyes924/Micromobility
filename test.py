import pandas as pd

hourly = pd.read_csv("data/processed/panels/hourly_24-25.csv")

print(hourly.info())