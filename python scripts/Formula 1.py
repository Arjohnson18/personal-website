# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:46:39 2025

@author: Arjoh
"""

import pandas as pd
from urllib.request import urlopen
import json

###--------------Import and Format Data 
#All data derived from https://openf1.org/v1/
#There are several importable points
    #For car data: /car_data
    #For drivers: /drivers
    #For intervals between driver and leader: /intervals
    #For individual laps: /laps
    #For meetings: /meetings
    #For pit stops: /pit
    #For position: /position
    #For race controls: /race_control
    #For sessions: /sessions
    #For stints: /stints
    #For weather: /weather
    
###--------------Examples
#To get all sessions in September 2023
response = urlopen('https://api.openf1.org/v1/sessions?date_start>=2023-09-01&date_end<=2023-09-30')
data = json.loads(response.read().decode('utf-8'))
df = pd.DataFrame(data)

#To fetch pit-out laps for driver number 55 (Carlos Sainz) that last at least 2 minutes 
response = urlopen('https://api.openf1.org/v1/laps?session_key=9222&driver_number=55&is_pit_out_lap=true&lap_duration>=120')
data = json.loads(response.read().decode('utf-8'))
df = pd.DataFrame(data)

#To get all sessions for the year 2023 in CSV format
response = urlopen('https://api.openf1.org/v1/sessions?year=2023&csv=true')

# Save to a file
with open("sessions_2023.csv", "w", encoding="utf-8") as file:
    file.write(response)

print("CSV file saved as sessions_2023.csv")

###--------------Print options
print(data)
print(df)
print(response)
    
###--------------Save Options    
#To save a DataFrame, use OBJname.to_csv
OBJname.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\OBJname.csv', mode='a') # mode =’a’ appends

import urllib.request

url = "https://api.openf1.org/v1/sessions?year=2023&csv=true"

    