
MetroBikeShare Stations Info:
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   Kiosk ID      441 non-null    int64
 1   Kiosk Name    441 non-null    object
 2   Go Live Date  441 non-null    object
 3   Region        439 non-null    object
 4   Status        441 non-null    object
 5   Latitude      441 non-null    float64
 6   Longitude     441 non-null    float64

Metro Trips Info:
 Data columns (total 15 columns):
 #   Column               Non-Null Count   Dtype
---  ------               --------------   -----
 0   trip_id              119865 non-null  int64
 1   duration             119865 non-null  int64
 2   start_time           119865 non-null  object
 3   end_time             119865 non-null  object
 4   start_station        119865 non-null  int64
 5   start_lat            119865 non-null  float64
 6   start_lon            119865 non-null  float64
 7   end_station          119865 non-null  int64
 8   end_lat              117304 non-null  float64
 9   end_lon              117304 non-null  float64
 10  bike_id              119865 non-null  int64
 11  plan_duration        119865 non-null  int64
 12  trip_route_category  119865 non-null  object
 13  passholder_type      119865 non-null  object
 14  bike_type            119865 non-null  object