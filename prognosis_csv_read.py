# Read prognoser.csv (as created by ../Spotprisprognose/data_extract.sh) using pandas
# and plot the prognosis for the next 24 hours.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

# Read the csv file
df = pd.read_csv('../data/prognoser.csv', sep=',', header=0, index_col=0, parse_dates=True)

# Substract DK2 and disregard 2 rows of different data format
df = df['DK2']
df = df.iloc[2:]
df
df[0]
type(df[0])
dict(df[0])

# Convert to PTime Atime value