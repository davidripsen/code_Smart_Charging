 # Read prognoser.csv (as created by ../Spotprisprognose/data_extract.sh) using pandas
# and plot the prognosis for the next 24 hours.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# Read the csv file
df = pd.read_csv('data/forecastsGreenerEl/prognoser.csv', sep=',', header=0, index_col=0, parse_dates=True)

# Substract DK2 and disregard 2 rows of different data format
df = df['DK2']
df = df.iloc[2:]
print(df)

# print df[2] nicely
print(df[2])
df[0]
type(df[0])
dict(df[0])

# Convert to PTime Atime value

#
for i in range(len(df)):
    print(f'Forecast created: ', df.index[i])
    print(df[i][:50])
    print(f'\n')
    #df[i] = dict(df[i])
    #df[i] = df[i]['PTime']
    #df[i] = dt.datetime.strptime(df[i], '%Y-%m-%dT%H:%M:%S')
    #df[i] = df[i].strftime('%H:%M')
