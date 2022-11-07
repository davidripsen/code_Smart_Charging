""" 
    Read and pre-process and visualise EV user data from Monta
"""

# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt
import plotly.graph_objects as go
import json

# Read EV user data
df = pd.read_csv('data/Monta/2022_11_03 10_15.csv', sep=',', header=0, parse_dates=True) # All charges
df2 = pd.read_csv('data/Monta/charge_smart_charges.csv', sep=',', header=0, parse_dates=True) # Smart Charges.    Can be mapped to All Charges on df2.id == df.Smart_Charge_ID

# Convert to datetime
timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'CHARGE_TIME']
for var in timevars:
    df[var] = pd.to_datetime(df[var], format='%Y-%m-%d %H:%M:%S')

timevars = ['start_time', 'stop_time', 'calculated_start_at', 'created_at', 'updated_at']
for var in timevars:
    df2[var] = pd.to_datetime(df2[var], format='%Y-%m-%d %H:%M:%S')

# Show df
df
print("Columns: ", df.columns)
df.iloc[5]

# Show df2
df2
print("Columns: ", df2.columns)
df2.iloc[43]


###################################################################################
######### NOW BACK TO df   ########################################################
###################################################################################
# USER_ID
print("Number of different users: ", df['USER_ID'].unique().size)
print("Number of transactions per unique user: "); df['USER_ID'].value_counts()

# Make plotly hist of df.USER_ID.value_counts()
fig = go.Figure(data=[go.Histogram(x=df['USER_ID'].value_counts())])
fig.update_layout(
    title_text="Number of transactions per unique user", # title of plot
    xaxis_title_text='Number of transactions', # xaxis label
    yaxis_title_text='Number of users', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()
df['USER_ID'].value_counts().describe()
df['USER_ID'].value_counts().median()


# Plot diurnal pattern of plug-in and plug-out times
def plot_hourly_plugtimes(df, var, title, xlab, ylab):
    fig = go.Figure(data=[go.Histogram(x=df[var].dt.hour)])
    fig.update_layout(
        title_text=title, # title of plot
        xaxis_title_text=xlab, # xaxis label
        yaxis_title_text=ylab, # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.show()

plot_hourly_plugtimes(df, 'CABLE_PLUGGED_IN_AT', "Daily pattern of CABLE_PLUGGED_IN_AT", "Hour of day", "Counts")
plot_hourly_plugtimes(df, 'RELEASED_AT', "Daily pattern of RELEASED_AT", "Hour of day", "Counts")

# Plot weekly pattern of plug-in and plug-out times
def plot_weekly_plugtimes(df, var, title, xlab, ylab):
    fig = go.Figure(data=[go.Histogram(x=df[var].dt.weekday)])
    fig.update_layout(
        title_text=title, # title of plot
        xaxis_title_text=xlab, # xaxis label
        yaxis_title_text=ylab, # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    # Change x-ticks to weekday names
    fig.update_xaxes(ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], tickvals=[0, 1, 2, 3, 4, 5, 6])
    fig.show()

plot_weekly_plugtimes(df, 'CABLE_PLUGGED_IN_AT', "Weekly pattern of CABLE_PLUGGED_IN_AT", "Day of week", "Counts")
plot_weekly_plugtimes(df, 'RELEASED_AT', "Weekly pattern of RELEASED_AT", "Day of week", "Counts")

# Plot diurnal-weekly pattern of plug-in and plug-out times
def plot_diurnal_weekly_plugtimes(df, var, title, xlab, ylab):
    fig = go.Figure(data=[go.Histogram2d(y=df[var].dt.hour, x=df[var].dt.weekday)])
    fig.update_layout(
        title_text=title, # title of plot
        xaxis_title_text=xlab, # xaxis label
        yaxis_title_text=ylab, # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    # Change x-ticks to weekday names
    fig.update_xaxes(ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], tickvals=[0, 1, 2, 3, 4, 5, 6])
    fig.show()

plot_diurnal_weekly_plugtimes(df, 'CABLE_PLUGGED_IN_AT', "Diurnal-weekly pattern of CABLE_PLUGGED_IN_AT", "Day of week", "Hour of day")
plot_diurnal_weekly_plugtimes(df, 'RELEASED_AT', "Diurnal-weekly pattern of RELEASED_AT", "Day of week", "Hour of day")

# Histogram of df['KWH']
fig = go.Figure(data=[go.Histogram(x=df['KWH'])])
fig.update_layout(
    title_text="Histogram of KWH", # title of plot
    xaxis_title_text='KWH', # xaxis label
    yaxis_title_text='Counts', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()


# For USER_ID = 2872, plot whole time serires of KWH using plotly
df[df['USER_ID'] == 2872].plot(x='CABLE_PLUGGED_IN_AT', y='KWH', kind='line', title='KWH for USER_ID = 2872')
plt.show()




###################################################################################
########### NOW BACK TO df2  ########################################################
###################################################################################
# NO unique USER_ID in df2
# only unique transaction id
print("Number of different transactions: ", df2['id'].unique().size)  
df2['priority_co2'].value_counts()
df2['priority_renewable'].value_counts()
df2['priority_price'].value_counts()
# Proportion choosing pure price priority
print("Proportion of customers choosing 100% price priority:      ", (df2['priority_price']==100).mean()) # 2/3 charge only for price
print("Proportion of customers choosing atleast 80% price priority:      ", (df2['priority_price']>=80).mean()) # 2/3 charge only for price

# Make 3D scatterplot of df2 and add jitter to x, y, z so that we can see the density
fig = go.Figure(data=[go.Scatter3d(
    z=df2['priority_price'] + np.random.normal(0, 3, df2['priority_price'].size),
    y=df2['priority_co2'] + np.random.normal(0, 3, df2['priority_co2'].size),
    x=df2['priority_renewable'] + np.random.normal(0, 3, df2['priority_renewable'].size),
    mode='markers',
    marker=dict(
        size=3,
        color=df2['priority_price'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.1
    )
)])
fig.update_layout(
    title_text="Smart Charge Priorities       " + str(round((df2['priority_price']==100).mean()*100,2)) + "% charge ONLY for price", # title of plot
    scene = dict(
        zaxis_title_text='Price', # xaxis label
        yaxis_title_text='CO2', # yaxis label
        xaxis_title_text='Renewable', # zaxis label
    ),
)
fig.show()

df2['priority_price'].describe()


# Plot daily, weekly and daily-weekly pattern of start_time and stop_time
plot_hourly_plugtimes(df2, 'start_time', "Daily pattern of start_time", "Hour of day", "Counts")
plot_hourly_plugtimes(df2, 'stop_time', "Daily pattern of stop_time", "Hour of day", "Counts")
plot_weekly_plugtimes(df2, 'start_time', "Weekly pattern of start_time", "Day of week", "Counts")
plot_weekly_plugtimes(df2, 'stop_time', "Weekly pattern of stop_time", "Day of week", "Counts")
plot_diurnal_weekly_plugtimes(df2, 'start_time', "Diurnal-weekly pattern of start_time", "Day of week", "Hour of day")
plot_diurnal_weekly_plugtimes(df2, 'stop_time', "Diurnal-weekly pattern of stop_time", "Day of week", "Hour of day")

# Example of results of Smart Charge
eval(df2.iloc[43].result)






##### Join df and df2 where df2.id == df.SMART_CHARGE_ID
#df2.id.isin(df.SMART_CHARGE_ID).mean() # The majority of Smart Charges are also in df (All Charges)
D = df2.merge(df, left_on='id', right_on='SMART_CHARGE_ID', how='left')
D = D.dropna(subset=['CABLE_PLUGGED_IN_AT'])

##### Let's examine a particular session
i = 3043 # i=43
sesh = D.iloc[i]
    # Plugged in at 02:04 and plugged-out at 12:08
    # Charged from 02:09 to 04:00
    # Charged 11 kWh at a maximum capacity of 11 kW (not possible within two hours)
eval(sesh.result)
    # Payed 41.2 kr. (2767 kg CO2) for
L = len(eval(sesh.result)['periods'])
xt = [eval(sesh.result)['periods'][i]['kwh'] for i in range(L)] # Charged
xt

print("kWh charged:  ", sum(xt))
prices = pd.DataFrame(eval(sesh.prices))
# Confirming total price:
#prices['value'].dot(xt)

# Hmmm.... result and df doesn't match
print("kWH  from df: ", sesh['KWH'], "kWh", " vs. ", "from df2: ", eval(sesh.result)['periods'][0]['kwh'] + eval(sesh.result)['periods'][1]['kwh'], "kWh")
print("CO2  from df: ", sesh['CO2'], "kg", " vs. ", "from df2: ", eval(sesh.result)['cost']['total_co2'], "kg")

sesh

########################### About the data ##########################################
##### What do we have:-)
# - Plug-in and plug-out times
# - kWh chaged
# - Unique user id to make patterns
# - kWh per specific hour charged


#### What do we not have:-(
# - Forecasts of price, CO2 and renewable (first is incoming - from Carnot.AI)


# Questions about to data:
# 1. UTC?
# 2. stop_time