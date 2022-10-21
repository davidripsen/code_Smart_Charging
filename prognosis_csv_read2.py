"""
READ and evaluate the forecasts as provided by email from Greener El / HMJ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime as dt

# Read the csv file
df = pd.read_csv('data/forecastsGreenerEl/forecasts_send_HMJ.csv', sep=';', header=0, parse_dates=True)    
df

# Convert HourUTC and HourFC to datetime
df['HourUTC'] = pd.to_datetime(df['HourUTC'], format='%Y-%m-%d %H:%M:%S')
df['HourFC'] = pd.to_datetime(df['HourFC'], format='%Y-%m-%d %H:%M:%S')
type(df['HourUTC'])

# Plot the data using plotly
fig = px.line(df, x='HourUTC', y='DK2SpotPriceEUR', title='Spot price')
fig.show()


# Count number of each unique instance of HourFC and print them all nicely
print(df['HourFC'].value_counts().to_string())
    # Der er super stor forskel på længden af de forskellige forecasts.
    #  + hvor mange gange han er kommet til at køre modellen / hvor mange modeller han har kørende.
    # Overvej bare at brug dem fra GitHub.

df.FC.unique()