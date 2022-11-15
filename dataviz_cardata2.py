""" 
    Read and pre-process 2nd upload of charge files and visualise EV user data from Monta
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import plotly.graph_objects as go
import datetime
pd.set_option('display.max_rows', 500)

#dfA1 = pd.read_csv('data/Monta/charges_part1.csv', sep=',', header=0, parse_dates=True, low_memory=False)
#dfA2 = pd.read_csv('data/Monta/charges_part2.csv', sep=',', header=0, parse_dates=True, low_memory=False)
#dfA3 = pd.read_csv('data/Monta/charges_part3.csv', sep=',', header=0, parse_dates=True, low_memory=False)
#D = pd.concat([dfA1, dfA2, dfA3], ignore_index=True)
dfB = pd.read_csv('data/Monta/vehicles.csv', header=0, index_col=None)
dfB.index = dfB.index +1 # VEHICLE_ID is index but starts at 1.
dfC1 = pd.read_csv('data/Monta/charges_extract_1.csv', header=0, parse_dates=True, low_memory=False)
dfC2 = pd.read_csv('data/Monta/charges_extract_2.csv', header=0, parse_dates=True, low_memory=False)
dfC3 = pd.read_csv('data/Monta/charges_extract_3.csv', header=0, parse_dates=True, low_memory=False)
spot = pd.read_csv('data/spotprice/df_spot_since_sept22.csv')
D = pd.concat([dfC1, dfC2, dfC3], ignore_index=True)
del dfC1, dfC2, dfC3

# Join dfB and D on dfb['id'] = D['vehicle_id']
D = D.merge(dfB, left_on='VEHICLE_ID', right_on=dfB.index, how='left')

# Convert to datetime and Copenhagen Time
timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'PLANNED_PICKUP_AT', 'ESTIMATED_COMPLETED_AT', 'LAST_START_ATTEMPT_AT','CHARGING_AT','STOPPED_AT']
for var in timevars:
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')
    # Convert from UTC to Copenhagen Time
    D[var] = D[var].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')

df_spot = pd.DataFrame({'time': spot['HourUTC'], 'price2': spot['SpotPriceDKK']/1000})
df_spot['time'] = pd.to_datetime(df_spot['time'], format='%Y-%m-%d %H:%M:%S')
df_spot = df_spot.set_index('time')
df_spot.index = df_spot.index.tz_localize('UTC').tz_convert('Europe/Copenhagen')


# Show
D.info()
D.iloc[44]

# Let's take a look at the data where SOC is available AND we are SMART CHARGING
D2 = D[D['SOC'].notna()]
D2 = D2[D2['SMART_CHARGE_ID'].notna()]
print("Disregarding ", round((1-(len(D2)/len(D))),4) *100, " % of the data where SOC is not available or we are not smart charging")
i = 2000

sesh = D2.iloc[i]
sesh

eval(sesh.KWHS)

sesh.KWHS
xt = pd.DataFrame(eval(sesh.KWHS))
prices = pd.DataFrame(eval(sesh.SPOT_PRICES))

# xt dot prices where time is the same
xt['time'] = pd.to_datetime(xt['time'])
prices['time'] = pd.to_datetime(prices['time'])
xt = xt.set_index('time')
prices = prices.set_index('time')

assert sesh.COST == round((xt * prices).sum()[0],4), "Costs do not match"
assert sesh.KWH == round(xt.sum()[0],4), "KWHs do not match"

print("SOC_START = ", sesh.SOC_START, "     SOC = ", sesh.SOC)

print("Number of different users:  ", D2.USER_ID.unique().shape[0])
print("Number of different vehicles:  ", D2.VEHICLE_ID.unique().shape[0])
print("Number of different smart chard IDs:  ", D2.SMART_CHARGE_ID.unique().shape[0])



############# Let's extract a single VEHICLE profile ########################################
vehicle_ids = D2.VEHICLE_ID.unique()
def PlotChargingProfile(D2, var="VEHICLE_ID", id=13267, vertical_hover=False):
    D2v = D2[D2[var] == id]
    D2v = D2v.sort_values(by=['CABLE_PLUGGED_IN_AT'])
    id = int(id)

    firsttime = D2v['CABLE_PLUGGED_IN_AT'].min().date() - datetime.timedelta(days=1)
    lasttime = D2v['PLANNED_PICKUP_AT'].max().date() + datetime.timedelta(days=1)

    # fig = go.Figure(data=[go.Scatter(
    #     x=D2v['CABLE_PLUGGED_IN_AT'],
    #     y=D2v['KWH'],
    #     mode='lines+markers',
    #     name='Plug-in',
    #     marker=dict(
    #         size=10,
    #         opacity=0.8
    #     )
    # )])

    # fig.add_trace(go.Scatter(
    #     x=D2v['PLANNED_PICKUP_AT'],
    #     y=D2v['KWH'],
    #     mode='lines+markers',
    #     name = "Plug-out",
    #     marker=dict(
    #         size=10,
    #         opacity=0.8,
    #         color='purple'
    #     )
    # ))

    # Create a list of times from firsttime to lasttime
    times = pd.date_range(firsttime, lasttime, freq='1h')
    # Create a list of zeros
    zeros = np.zeros(len(times))
    nans = np.full(len(times), np.nan)
    # Create a dataframe with these times and zeros
    df = pd.DataFrame({'time': times, 'z_plan': zeros, 'z_act': zeros, 'charge': zeros, 'price': nans, 'SOC': nans})
    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
    df.z_plan, df.z_act = -1, -1
    # Set the index to be the time
    df = df.set_index('time')
    
    # Loop over all plug-ins and plug-outs     # ADD KWH AND SOC RELATIVE TO TIMES
    for i in range(len(D2v)):
        # Set z=1 for all times from plug-in to plug-out
        df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['PLANNED_PICKUP_AT'], 'z_plan'] = 1
        df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['RELEASED_AT'], 'z_act'] = 1
        # Extract charge from 'KWHS' and add to df where time is the same
        xt = pd.DataFrame(eval(D2v.iloc[i]['KWHS']))
        if D2v.iloc[i]['KWH'] != round(xt.sum()[1],4):
            print("KWH total and sum(kWh_t) does not match for D2v row i=", i)
        xt['time'] = pd.to_datetime(xt['time'])
        xt['time'] = xt['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
        xt = xt.set_index('time')
        df.loc[xt.index, 'charge'] = xt['value']

        # Add the right spot prices to df
        if type(D2v.iloc[i]['SPOT_PRICES']) == str:
            prices = pd.DataFrame(eval(D2v.iloc[i]['SPOT_PRICES']))
            prices['time'] = pd.to_datetime(prices['time'])
            prices['time'] = prices['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
            prices = prices.set_index('time')
            df.loc[prices.index, 'price'] = prices['value']
        
        # Add SOC and convert to kWhs
        if show_SOC:
            df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT'].ceil('H'), 'SOC'] = D2v.iloc[i]['SOC_START']/100 * D2v.iloc[i]['capacity_kwh']
            df.loc[D2v.iloc[i]['PLANNED_PICKUP_AT'].floor('H'), 'SOC'] = D2v.iloc[i]['SOC']/100 * D2v.iloc[i]['capacity_kwh']

    df['costs'] = df['price'] * df['charge']

    df = df.merge(df_spot, how='left', left_on='time', right_on='time')
    df['BatteryCapacity'] = D2v.iloc[i]['capacity_kwh']
    df['CableCapacity'] = D2v.iloc[i]['max_kw_dc']
    
    # in df['SOC] replace nan with most recent value
    df['SOC'] = df['SOC'].fillna(method='ffill')


    fig = go.Figure([go.Scatter(
    x=df.index,
    y=df['z_act'],
    mode='lines',
    name = "Plugged-in (actual) [true/false]",
    line=dict(
        color='darkblue'
    ))])

    # Plot the result
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['z_plan'],
        mode='lines',
        name='Plugged-in (planned) [true/false]',
        line=dict(
            color='black',
    )))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['charge'],
        mode='lines',
        name='Charge [kWh]',
        marker=dict(
            size=10,
            opacity=0.8
        ),
        line=dict(
            color='red',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name='Price [DKK/kWh excl. tarifs]',
        line=dict(
            color='purple',
            width=1
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price2'],
        mode='lines',
        name='Price2 (loaded from EnergiDataService.dk) [DKK/kWh excl. tarifs]',
        line=dict(
            color='purple',
            width=1,
            dash='dot'
        )
    ))

    # fig.add_trace(go.Scatter(
    #     x=D2v['CABLE_PLUGGED_IN_AT'],
    #     y=D2v['SOC_START'],
    #     mode='lines+markers',
    #     name = "SOC_start",
    #     marker=dict(
    #         size=5,
    #         opacity=0.3,
    #         color='navy'
    #     ),
    #     line=dict(width=0.5, color='DarkSlateGrey')
    # ))

    # fig.add_trace(go.Scatter(
    #     x=D2v['PLANNED_PICKUP_AT'],
    #     y=D2v['SOC'],
    #     mode='lines+markers',
    #     name = "SOC_end",
    #     marker=dict(
    #         size=5,
    #         opacity=0.3,
    #         color='darkblue'
    #     ),
    #     line=dict(width=0.5, color='DarkSlateGrey')

    # ))

    fig.add_trace(go.Scatter(
        x=D2v['PLANNED_PICKUP_AT'],
        y=D2v['SOC_LIMIT']/100 * D2v['capacity_kwh'],
        mode='lines+markers',
        name = "SOC limit",
        marker=dict(
            size=3,
            opacity=0.3,
            color='darkgrey'
        )
        # Add index value to hovertext
        # hovertext = df.index
        
    ))

    # Show BatteryCapacity
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BatteryCapacity'],
        mode='lines',
        name = "Battery Capacity",
        line=dict(
            color='darkgrey',
            dash='dash'
    )))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SOC'],
        mode='lines',
        name = "SOC",
        line=dict(
            color='lightblue'
        )
    ))

    if vertical_hover:
        # Add vertical hover lines
        fig.update_layout(
            hovermode='x unified',
            hoverdistance=100, # Distance to show hover label of data point
            spikedistance=1000, # Distance to show spike
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
            
    # Set xticks to be individual days
    fig.update_xaxes(
        tickmode = 'array',
        tickvals = [firsttime + datetime.timedelta(days=i) for i in range((lasttime-firsttime).days+1)],
        ticktext = [str(firsttime + datetime.timedelta(days=i))[:10] for i in range((lasttime-firsttime).days+1)],
        tickangle = 45
    )
    # Add legend
    fig.update_layout(
        title_text="Charging by " +str(var) + "="+ str(id) + "               from "+str(firsttime)+"    to   "+str(lasttime), # title of plot
        xaxis_title_text="Date", # xaxis label
        yaxis_title_text="kWh or DKK/kWh", # yaxis label
        #font=dict(
        #    size=18,
        #    color="RebeccaPurple"
        #)
    )
    fig.show()
    return df
dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=vehicle_ids[4], show_SOC=True, vertical_hover=True)

ids = np.random.choice(vehicle_ids, 5, replace=False)
for id in ids:
    print("Plotting vehicle", id)
    PlotChargingProfile(D2, id=id)



# Get data from VEHICLE_ID == vid
D2v = D2[D2['VEHICLE_ID'] == 31920]
D2v.CABLE_PLUGGED_IN_AT
sesh = D2v.loc[453865]
eval(sesh.KWHS)
sesh.KWH

sesh.CABLE_PLUGGED_IN_AT
sesh.PLANNED_PICKUP_AT


############## What I need - how I need it  ###############################################
# z_t possible to calculate in near-future  :-)  z_t = CABLE_PLUGGED_IN - RELEASED_AT (eller PLANNED_PICKUP_AT ?), when Marcos has extracted as datetime
# u_t   = kan udregnes fra SOC (eller omvendt)
# b_t (SOC) = ???? (De få SOC-værdier, der er, ser ikke ud til at være korrekte)
#   - Det er vist ikke rigtig muligt at lave noget Multi-Day Smart Charge uden.
#   - De bruger den jo nok ikke, når de kun kigger frem til plug-out. Og de lader ekstra, hvis prisen er meget lav.
#   - Det er ikke faktisk ikke så tosset en idé bare at lade ekstra, når forecast prisen er høj de næste dage.
#   - Men er det egentlig ikke lidt det samme som implicit og ubevidst at forecaste u_t og z_t ?
# MONTA bruger KUN forecasts, hvis PLANNED_PICKUP_AT er senere end hvor vi har kendte SPOT_PRICES.

# bmax (burde være der et sted) = KWH_LIMIT    (SOC_LIMIT = 80 or 100 % (presumably set by user))
# xmax = now_kwh (charge_smart_charges.csv) :-)

# Decision var (as they do it)
# xt = xt                                   :-)

##### Data Quality
# 1. SOC is not always correct. For vehicle_ids[4] it does not change after during or after charging.
# 2. SOC can increase in a sesh, eventhough no charging is performed. Here, charging says 0 KWH, but there has been some charging (MORE THAN THE SOC INCREASE),
#       and the charging was performed a week LATER than the plug-in:
#     > D2v = D2[D2['VEHICLE_ID'] == 31572];
#     > sesh = D2v.loc[184069];
#     > eval(sesh.KWHS)
#
# VEHICLES / battery_cap is now well-matched:-)