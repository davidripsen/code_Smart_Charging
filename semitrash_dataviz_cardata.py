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
import datetime
pd.set_option('display.max_rows', 500)
layout = dict(font=dict(family='Computer Modern',size=11),
              margin=dict(l=5, r=5, t=30, b=5),
              width=605, height= 250, title_x = 0.5)
path = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/EV_Monta/Individual_EVs/'
pathhtml = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/_figures/'


# # Read EV user data
# df = pd.read_csv('data/Monta/2022_11_03 10_15.csv', sep=',', header=0, parse_dates=True) # All charges
df2 = pd.read_csv('data/Monta/charge_smart_charges.csv', sep=',', header=0, parse_dates=True) # Smart Charges.    Can be mapped to All Charges on df2.id == df.Smart_Charge_ID
# df_vis = pd.read_excel('data/Monta/core.core_charges.xlsx')
# #dfA1 = pd.read_csv('data/Monta/charges_part1.csv', sep=',', header=0, parse_dates=True, low_memory=False)
# #dfA2 = pd.read_csv('data/Monta/charges_part2.csv', sep=',', header=0, parse_dates=True, low_memory=False)
# #dfA3 = pd.read_csv('data/Monta/charges_part3.csv', sep=',', header=0, parse_dates=True, low_memory=False)
# # A2, A1, A3
# #df3 = pd.concat([dfA1, dfA2, dfA3], ignore_index=True)


# # Convert to datetime
# timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'CHARGE_TIME']
# for var in timevars:
#     df[var] = pd.to_datetime(df[var], format='%Y-%m-%d %H:%M:%S')

# timevars = ['start_time', 'stop_time', 'calculated_start_at', 'created_at', 'updated_at']
# for var in timevars:
#     df2[var] = pd.to_datetime(df2[var], format='%Y-%m-%d %H:%M:%S')

# # Show df
# df
# print("Columns: ", df.columns)
# df.iloc[5]

# # Show df2
# df2
# print("Columns: ", df2.columns)
# df2.iloc[43]

# ##### Join df and df2 where df2.id == df.SMART_CHARGE_ID
#     #df2.id.isin(df.SMART_CHARGE_ID).mean() # The majority of Smart Charges are also in df (All Charges)
# D = df2.merge(df, left_on='id', right_on='SMART_CHARGE_ID', how='left')
# D = D.dropna(subset=['CABLE_PLUGGED_IN_AT'])
# D

# # Show df3
# df3
# df3.iloc[44]


dfB = pd.read_csv('data/Monta/vehicles.csv', header=0, index_col=None)
dfB.index = dfB.index +1 # VEHICLE_ID is index but starts at 1.
dfC1 = pd.read_csv('data/Monta/charges_extract_1.csv', header=0, parse_dates=True, low_memory=False)
dfC2 = pd.read_csv('data/Monta/charges_extract_2.csv', header=0, parse_dates=True, low_memory=False)
dfC3 = pd.read_csv('data/Monta/charges_extract_3.csv', header=0, parse_dates=True, low_memory=False)
spot = pd.read_csv('data/spotprice/df_spot_2022.csv')
D = pd.concat([dfC1, dfC2, dfC3], ignore_index=True)
#del dfC1, dfC2, dfC3

# Join dfB and D on dfb['id'] = D['vehicle_id']
D = D.merge(dfB, left_on='VEHICLE_ID', right_on=dfB.index, how='left')

# Convert to datetime and Copenhagen Time
timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'PLANNED_PICKUP_AT', 'ESTIMATED_COMPLETED_AT', 'LAST_START_ATTEMPT_AT','CHARGING_AT','STOPPED_AT']
for var in timevars:
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')
    # Convert from UTC to Copenhagen Time
    D[var] = D[var].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')

df_spot = pd.DataFrame({'time': spot['HourUTC'], 'trueprice': spot['SpotPriceDKK']/1000})
df_spot['time'] = pd.to_datetime(df_spot['time'], format='%Y-%m-%d %H:%M:%S')
df_spot = df_spot.set_index('time')
df_spot.index = df_spot.index.tz_localize('UTC').tz_convert('Europe/Copenhagen')

# Let's take a look at the data where SOC is available AND we are SMART CHARGING
D2 = D[D['SOC'].notna()]
df = D2[D2['SMART_CHARGE_ID'].notna()]
print("Disregarding ", round((1-(len(D2)/len(D))),4) *100, " % of the data where SOC is not available or we are not smart charging")
df.iloc[44]


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
def plot_diurnal_weekly_plugtimes(df, var, title, xlab, ylab, export=False):
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

    # Export figure
    if export:
        fig.write_html( "plots/EV_Monta/diurnal_weekly_" + var + ".html")
        fig.update_layout(layout)
        fig.write_image("plots/EV_Monta/diurnal_weekly_" + var + ".pdf")

plot_diurnal_weekly_plugtimes(df, 'CABLE_PLUGGED_IN_AT', "Daily-weekly pattern of CABLE_PLUGGED_IN_AT by SmartCharge users", "Day of week", "Hour of day", export=True)
plot_diurnal_weekly_plugtimes(df, 'PLANNED_PICKUP_AT', "Daily-weekly pattern of PLANNED_PICKUP_AT by SmartCharge users", "Day of week", "Hour of day", export=True)
plot_diurnal_weekly_plugtimes(df, 'RELEASED_AT', "Daily-weekly pattern of RELEASED_AT by SmartCharge users", "Day of week", "Hour of day", export=True)

# Histogram of df['KWH']
fig = go.Figure(data=[go.Histogram(x=df['KWH'])])
fig.update_layout(
    title_text="Histogram of KWH", # title of plot
    xaxis_title_text='KWH', # xaxisDabel
    yaxis_title_text='Counts', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.show()


# For USER_ID = 2872, plot whole time serires of KWH using plotly
df[df['USER_ID'] == 2872].plot(x='CABLE_PLUGGED_IN_AT', y='KWH', kind='line', title='KWH for USER_ID = 2872')
plt.show()




####################################################################################
########### NOW BACK TO df2  #######################################################
####################################################################################
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
# Export figure
fig.write_html( "plots/EV_Monta/3D_scatterplot_priorities.html")

df2['priority_price'].describe() # 82 % average for price


# Plot daily, weekly and daily-weekly pattern of start_time and stop_time
plot_hourly_plugtimes(df2, 'start_time', "Daily pattern of start_time", "Hour of day", "Counts")
plot_hourly_plugtimes(df2, 'stop_time', "Daily pattern of stop_time", "Hour of day", "Counts")
plot_weekly_plugtimes(df2, 'start_time', "Weekly pattern of start_time", "Day of week", "Counts")
plot_weekly_plugtimes(df2, 'stop_time', "Weekly pattern of stop_time", "Day of week", "Counts")
plot_diurnal_weekly_plugtimes(df2, 'start_time', "Diurnal-weekly pattern of start_time", "Day of week", "Hour of day")
plot_diurnal_weekly_plugtimes(df2, 'stop_time', "Diurnal-weekly pattern of stop_time", "Day of week", "Hour of day")

# Example of results of Smart Charge
eval(df2.iloc[43].result)



###################################################################################
########### NOW BACK TO D  ########################################################
###################################################################################

######## Let's examine a particular session #######################################
i = 3543 # i=43
sesh = D.iloc[i]
sesh
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
prices['value'].dot(xt)

# Hmmm.... result and df doesn't match
print("kWH  from df: ", sesh['KWH'], "kWh", " vs. ", "from df2: ", eval(sesh.result)['periods'][0]['kwh'] + eval(sesh.result)['periods'][1]['kwh'], "kWh")
print("CO2  from df: ", sesh['CO2'], "kg", " vs. ", "from df2: ", eval(sesh.result)['cost']['total_co2'], "kg")



###### Let's examine a particular user ############################################
user = 43456
# For the specific user, plot 'KWH' vs. 'CABLE_PLUGGED_IN_AT' using plotly
D_user = D[D['USER_ID'] == user]
D_user = D_user.sort_values(by=['CABLE_PLUGGED_IN_AT'])

firsttime = D_user['CABLE_PLUGGED_IN_AT'].min()
lasttime = D_user['RELEASED_AT'].max()

fig = go.Figure(data=[go.Scatter(
    x=D_user['RELEASED_AT'],
    y=D_user['KWH'],
    mode='lines+markers',
    marker=dict(
        size=10,
        opacity=0.8
    )
)])
# Set xticks to be individual days
fig.update_xaxes(
    tickmode = 'array',
    tickvals = [firsttime + datetime.timedelta(days=i) for i in range((lasttime-firsttime).days+1)],
    ticktext = [str(firsttime + datetime.timedelta(days=i))[:10] for i in range((lasttime-firsttime).days+1)],
    tickangle = 45
) 
fig.update_layout(
    title_text="Charging by user " + str(user) + "               from "+str(firsttime.date())+" to "+str(lasttime.date()), # title of plot
    xaxis_title_text="Date", # xaxis label
    yaxis_title_text="KWH", # yaxis label
    legend_title="Legend Title", # title for legend
    #font=dict(
    #    size=18,
    #    color="RebeccaPurple"
    #)
)
fig.show()


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

##### Other comments
# - regard Charge-file. Smart_Charge-file corresponds to PLANNED SmartCharge. Often PLUGGED-OUT much before.
# - awaiting new df
# - awaiting new forecasts
# - until then: