""" 
    Read and pre-process 2nd upload of charge files and visualise EV user data from Monta
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
import pickle
pd.set_option('display.max_rows', 500)
layout = dict(font=dict(family='Computer Modern',size=11),
              margin=dict(l=5, r=5, t=30, b=5),
              width=605, height= 250,
              title_x = 0.5,
              legend=dict(orientation="h", yanchor="bottom", y=-.32, xanchor="right", x=1))
path = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/EV_Monta/Individual_EVs/'
pathhtml = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/_figures/'

D = pd.read_csv('data/Monta/testdata/SANDBOX.csv', header=0, parse_dates=True, low_memory=False)
spot = pd.read_csv('data/spotprice/df_spot_2022.csv')
dfB = pd.read_csv('data/Monta/vehicles.csv', header=0, index_col=None)
dfB.index = dfB.index +1 # VEHICLE_ID is index but starts at 1.
spot = pd.read_csv('data/spotprice/df_spot_2022.csv')
# Join dfB and D on dfb['id'] = D['vehicle_id']
D = D.merge(dfB, left_on='VEHICLE_ID', right_on=dfB.index, how='left')

# Convert to datetime and Copenhagen Time
timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'PLANNED_PICKUP_AT', 'ESTIMATED_COMPLETED_AT', 'LAST_START_ATTEMPT_AT','CHARGING_AT','STOPPED_AT']
for var in timevars:
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')
    # Convert from UTC to Copenhagen Time
    D[var] = D[var].dt.tz_convert('Europe/Copenhagen')
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')

df_spot = pd.DataFrame({'time': spot['HourUTC'], 'trueprice': spot['SpotPriceDKK']/1000})
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
i = 2
sesh = D2.iloc[i]
eval(sesh.KWHS)
print(sesh.to_latex(index=True))

# sesh.KWHS
# xt = pd.DataFrame(eval(sesh.KWHS))
# prices = pd.DataFrame(eval(sesh.SPOT_PRICES))

# # xt dot prices where time is the same
# xt['time'] = pd.to_datetime(xt['time'])
# prices['time'] = pd.to_datetime(prices['time'])
# xt = xt.set_index('time')
# prices = prices.set_index('time')

#assert sesh.COST == round((xt * prices).sum()[0],4), "Costs do not match"
#assert sesh.KWH == round(xt.sum()[0],4), "KWHs do not match"

print("SOC_START = ", sesh.SOC_START, "     SOC = ", sesh.SOC)

print("Number of different users:  ", D2.USER_ID.unique().shape[0])
print("Number of different vehicles:  ", D2.VEHICLE_ID.unique().shape[0])
print("Number of different smart chard IDs:  ", D2.SMART_CHARGE_ID.unique().shape[0])



############# Let's extract a single VEHICLE profile ########################################
vehicle_ids = D2.VEHICLE_ID.unique()
var = 'VEHICLE_ID'
dfvehicle=None
id = 13267
def PlotChargingProfile(D2=None, dfvehicle=None, var="VEHICLE_ID", id=13267, plot_efficiency_and_SOCmin=True, vertical_hover=False, df_only=False, layout=None, imgtitle="PlainProfile_id"):
    """
    Plot the charging profile of a single vehicle
    If df_only is True, then only the dataframe is returned
    If df_vehicle is not None, then only plotting is done
    """

    if dfvehicle is None:
        D2v = D2[D2[var] == id]
        D2v = D2v.sort_values(by=['CABLE_PLUGGED_IN_AT'])
        id = int(id)

        firsttime = D2v['CABLE_PLUGGED_IN_AT'].min().date() - datetime.timedelta(days=1)
        lasttime = max( D2v['PLANNED_PICKUP_AT'].max().date(), D2v['RELEASED_AT'].max().date()) + datetime.timedelta(days=1)

        assert len(D2v.capacity_kwh.unique()) == 1, "Battery capacity changes for vehicle " + str(id)
        assert len(D2v.max_kw_ac.unique()) == 1, "Cable capacity changes for vehicle " + str(id)

        # Create a list of times from firsttime to lasttime
        times = pd.date_range(firsttime, lasttime, freq='1h')
        # Create a list of zeros
        zeros = np.zeros(len(times))
        nans = np.full(len(times), np.nan)
        # Create a dataframe with these times and zeros
        df = pd.DataFrame({'time': times, 'z_plan': zeros, 'z_act': zeros, 'charge': zeros, 'price': nans, 'SOC': nans, 'SOCmin': nans, 'SOCmax': nans, 'BatteryCapacity': nans, 'CableCapacity': nans, 'efficiency': nans})
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
        df.z_plan, df.z_act = -1, -1
        # Set the index to be the time
        df = df.set_index('time')
        
        # Vehicle specifics
        df['BatteryCapacity'] = D2v.iloc[-1]['capacity_kwh']
        df['CableCapacity'] = D2v.iloc[-1]['max_kw_ac']

        # Loop over all plug-ins and plug-outs     # ADD KWH AND SOC RELATIVE TO TIMES
        for i in range(len(D2v)):
            # Set z=1 for all times from plug-in to plug-out
            df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['PLANNED_PICKUP_AT'], 'z_plan'] = 1 #i=2, ser ud til at være fucked, når CABLE_PLUGGED_IN_AT IKKE er heltal.
            df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['RELEASED_AT'], 'z_act'] = 1

            # Allow semi-discrete plug-in relative to proportion of the hour
            #df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT'], 'z_plan'] = 1

            # # Extract charge from 'KWHS' and add to df where time is the same
            # xt = pd.DataFrame(eval(D2v.iloc[i]['KWHS']))
            # if D2v.iloc[i]['KWH'] != round(xt.sum()[1],4):
            #     print("KWH total and sum(kWh_t) does not match for D2v row i=", i)
            # xt['time'] = pd.to_datetime(xt['time'])
            # xt['time'] = xt['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
            # xt = xt.set_index('time')
            # df.loc[xt.index, 'charge'] = xt['value']

            # Efficiency of charging (ratio of what has been charged to what goes into the battery)
            #if D2v.iloc[i]['KWH'] >= 1: # Only proper charging
            if D2v.iloc[i]['KWH'] > 0:
                df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['RELEASED_AT'], 'efficiency'] = ((D2v.iloc[i].SOC - D2v.iloc[i].SOC_START) / 100 * D2v.iloc[i]['capacity_kwh']) / D2v.iloc[i].KWH

            # # Add the right spot prices to df
            # if type(D2v.iloc[i]['SPOT_PRICES']) == str and len(eval(D2v.iloc[i]['SPOT_PRICES'])) != 0:
            #     prices = pd.DataFrame(eval(D2v.iloc[i]['SPOT_PRICES']))
            #     prices['time'] = pd.to_datetime(prices['time'])
            #     prices['time'] = prices['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
            #     prices = prices.set_index('time')
            #     df.loc[prices.index, 'price'] = prices['value']
            
            # Add SOC and convert to kWhs
            df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT'].ceil('H', ambiguous=bool), 'SOC'] = D2v.iloc[i]['SOC_START']/100 * D2v.iloc[i]['capacity_kwh']
            df.loc[D2v.iloc[i]['PLANNED_PICKUP_AT'].floor('H'), 'SOC'] = D2v.iloc[i]['SOC']/100 * D2v.iloc[i]['capacity_kwh']

            # Add SOCmax
            df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['PLANNED_PICKUP_AT'], 'SOCmax'] = D2v.iloc[i]['SOC_LIMIT']/100 * D2v.iloc[i]['capacity_kwh']

            # bmin (PURELY INPUT ASSUMPTION)
            min_charged = 0.40 # 40% of battery capacity
            min_alltime = 0.05 # Never go below 5%
            df.loc[D2v.iloc[i]['PLANNED_PICKUP_AT'].floor('H'), 'SOCmin'] = min_charged * df['BatteryCapacity'][i] # Min SOC
            df['SOCmin'] = df['SOCmin'].fillna(min_alltime * df['BatteryCapacity'][i])


        # If z_plan_everynight and corresponding bmin
        # z_plan_everynight:
        df['z_plan_everynight'] = -1 # df['z_plan_everynight'] = df['z_plan']
        df.loc[(df.index.hour >= 22) | (df.index.hour < 6), 'z_plan_everynight'] = 1

        # bmin_everymorning:
        df['SOCmin_everymorning'] = min_alltime * df['BatteryCapacity']  #df['SOCmin_everymorning'] = df['SOCmin']
        df.loc[(df.index.hour == 6), 'SOCmin_everymorning'] = min_charged * df['BatteryCapacity']

        # Costs
        df['costs'] = df['price'] * df['charge']
        df = df.merge(df_spot, how='left', left_on='time', right_on='time')
        
        # in df['SOC] replace nan with most recent value
        df['SOC_lin'] = df['SOC'].interpolate(method='linear')
        df['SOC'] = df['SOC'].fillna(method='ffill')

        # Use
        u = df.SOC.diff().dropna()
        u[u>0] = 0
        u = u.abs()
        df['use'] = u

        # Use linearly interpolated SOC
        u_lin = df.SOC_lin.diff().dropna()
        u_lin[u_lin>0] = 0
        u_lin = u_lin.abs()
        df['use_lin'] = u_lin
        # Daily average use
        df['use_dailyaverage'] = df[df['use_lin'] != 0]['use_lin'].mean()

        # Calculate 7-day rolling mean of use_lin
        roll_length = 7 # If changed, also change in legend
        df['use_rolling'] = df[df['use_lin'] != 0]['use_lin'].rolling(roll_length*24, min_periods=24).mean()
        df['use_rolling'] = df['use_rolling'].fillna(0)
        # Issues: When subsetting on NOT plugged_in, the roll length of 7*24 steps becomes more than 7 days
        # Issues: Initial 7 days

        # Calculate 14-day rolling mean of use (use to estimate use_lin. Without cheating)
        roll_length = 10
        df['use_org_rolling'] = df['use'].rolling(roll_length*24, min_periods=12).mean() # min periods shouldn't be too large or too small
        df['use_org_rolling'] = df['use_org_rolling'].fillna(0) # Estimate u_hat 12 hours with 0

        # Exponential moving average
        hlf_life = 2 # days
        df['use_ewm'] = df[df['use_lin'] != 0]['use_lin'].ewm(span=roll_length*24, min_periods=24).mean()
        df['use_ewm'] = df['use_ewm'].fillna(0)

        # Median prediction of efficiency
        df['efficiency_median'] = np.median(df['efficiency'].dropna().unique())

        # Add vehicle id
        df['vehicle_id'] = id

        # Assure non-Nan at crucial places
        if any(df['use'].isna()):
            df = df[~df['use_lin'].isna()]
            print('Rows with NaNs in Use were deleted.')

    else:
        df = dfvehicle
        firsttime = df.index[0]
        lasttime = df.index[-1]

    #################### START THE PLOTTING ###########################################
    fig = go.Figure([go.Scatter(
        x=df.index,
        y=df['z_act'],
        mode='lines',
        name = "Plugged-in (actual)",
        line=dict(
            color='black',
            dash='dot',
    ))])

    # Plot the result
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['z_plan'],
        mode='lines',
        name='Plugged-in (planned)',
        line=dict(
            color='black',
    )))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['z_plan_everynight'],
        mode='lines',
        name='Plugged-in (assumption)',
        line=dict(
            color='black',
    )))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['charge'],
        mode='lines',
        name='Charge',
        marker=dict(
            size=10,
            opacity=0.8
        ),
        line=dict(
            color='green',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['use'],
        mode='lines',
        name='Use',
        line=dict(
            color='red',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['use_lin'],
        mode='lines',
        name='Use (interpolated)',
        line=dict(
            color='red',
            width=2,
            dash='dot'
        )
    ))

    # fig.add_trace(go.Scatter(
    # x=df.index,
    # y=df['use_rolling'],
    # mode='lines',
    # name='Use ('+str(7)+' day rolling mean) [kWh]',
    # line=dict(
    #     color='red',
    #     width=2,
    #     dash='dot'
    # )
    # ))

    # fig.add_trace(go.Scatter(
    # x=df.index,
    # y=df['use_ewm'],
    # mode='lines',
    # name='Use (Exponentially Weighted Moving Average with half life = '+str(2)+') [kWh]',
    # line=dict(
    #     color='red',
    #     width=2,
    #     dash='dash'
    # )
    # ))

    # fig.add_trace(go.Scatter(
    #     x=df.index,
    #     y=df['use_dailyaverage'],
    #     mode='lines',
    #     name='Use daily average (outside of plug-in) [kWh]',
    #     line=dict(
    #         color='red',
    #         width=0.5,
    #         dash='dash'
    #     )
    # ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(
            color='purple',
            width=1
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SOC'],
        mode='lines',
        name = "SOC",
        line=dict(
            color='lightblue',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
    x=df.index,
    y=df['trueprice'],
    mode='lines',
    name='Price',
    line=dict(
        color='purple',
        width=1
    )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SOCmax'],
        mode='lines',
        name = "SOC max [kWh]",
        line=dict(width=2, color='grey') #color='DarkSlateGrey')
        # Add index value to hovertext
        # hovertext = df.index
    ))

    if plot_efficiency_and_SOCmin:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['efficiency']*100,
            mode='lines',
            name = "Efficiency [%]",
            line=dict(width=2, color='DarkSlateGrey')
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['efficiency_median']*100,
            mode='lines',
            name = "Efficiency median [%]",
            line=dict(width=2, color='DarkSlateGrey', dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
        y=df['SOC_lin'],
        mode='lines',
        name = "SOC (linear interpolation)",
        line=dict(
            color='lightblue',
            width=2,
            dash='dot'
        )
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SOCmin_everymorning'],
            mode='lines',
            name = "Minimum SOC",
            line=dict(
                color='lightblue',
                width=2 , dash='dash'
            )
            ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BatteryCapacity'],
        mode='lines',
        name = "Battery Capacity [kWh]",
        line=dict(
            color='darkgrey',
            dash='dash'
    )))

    fig.add_trace(go.Scatter(
    x=df.index,
    y=df['use_org_rolling'],
    mode='lines',
    name = "Rolling Use (10 days)",
    line=dict(
        color='red',
        width=1,
        dash='dash'
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
        yaxis_title_text="kWh or True-False [1, -1]", # yaxis label
        #font=dict(
        #    size=18,
        #    color="RebeccaPurple"
        #)
    )

    if layout is not None:
        # Export html
        fig.write_html(pathhtml + imgtitle + str(id) + ".html")
        fig.update_layout(layout)
        # For the x-ticks, only show every 7th day
        fig.update_xaxes(
            tickmode = 'array',
            tickvals = [firsttime + datetime.timedelta(days=i) for i in range((lasttime-firsttime).days+1) if i%7==0],
            ticktext = [str(firsttime + datetime.timedelta(days=i))[:10] for i in range((lasttime-firsttime).days+1) if i%7==0],
            tickangle = 45
        )
        # Remove x-ticks and xaxis title text (TEMPORARY)
        # fig.update_xaxes(
        #     showticklabels=False,
        #     title_text=""
        # )

        # Subset data to 2022-09-20 to 2022-09-30
        # fig.update_xaxes(
        #     range=['2022-09-20', '2022-09-30']
        # )
        # fig.update_yaxes(
        #     range=[-1, 10]
        # )

        # Decrease linewidth of all lines
        for i in range(len(fig.data)):
            fig.data[i].line.width = 1.5
        # Export pdf
        fig.write_image(path + imgtitle + str(id) + ".pdf")
        
    if not df_only:
        fig.show()
    return df
## Export plots for report
#dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=13923, plot_efficiency_and_SOCmin=True, vertical_hover=False, layout=layout, imgtitle="use_curves_id")

dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=853, plot_efficiency_and_SOCmin=True, vertical_hover=False)
dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=vehicle_ids[99], vertical_hover=False)
dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=24727, plot_efficiency_and_SOCmin=True, vertical_hover=False, df_only=False)
dfv = PlotChargingProfile(D2, var="VEHICLE_ID", id=14597, plot_efficiency_and_SOCmin=True, vertical_hover=False)
# Drop variables in dfv
dfv = dfv.drop(['use_dailyaverage', 'use_rolling', 'use_ewm', 'efficiency'], axis=1)
# Add index as the first column and reset index
# dfv.insert(0, 'time', dfv.index)
# dfv = dfv.reset_index(drop=True)
# # transpose dfv
# dfv = dfv.T
# print(round(dfv.iloc[:,[0,-1]],2).to_latex())
# print(round(dfp.iloc[[0,1,2,3,4,5,-3,-2,-1],[0,1,2,3,4,5,6,-3,-2,-1]],2).to_latex())


# ids = [30299, 6817, 18908] # Ids where perfect foresight fails
# for id in ids:
#     print("Plotting vehicle", id)
#     dfv = PlotChargingProfile(D2, id=id, plot_efficiency_and_SOCmin=True)


# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Extract vehicle ids from DFV
selected_vehicle_ids = []
for df in DFV:
    selected_vehicle_ids.append(df['vehicle_id'][0])

# Subset testdata
firsttime = datetime.datetime(2022, 11, 12, 0, 0, 0)
lasttime = datetime.datetime(2023, 1, 11, 11, 59, 59)
firsttime = firsttime.replace(tzinfo=datetime.timezone.utc)
lasttime = lasttime.replace(tzinfo=datetime.timezone.utc)

# Remove vehicles manually, if they practicically didn't charge in the testperiod
# Export
selected_vehicle_ids = [x for x in selected_vehicle_ids if x not in [3011, 1352, 24727, 14597, 32278, 21745]]
DFVtest = []
for id in selected_vehicle_ids:
    print("Plotting vehicle", id)
    dfv = PlotChargingProfile(D2, id=id, df_only=True, plot_efficiency_and_SOCmin=False, vertical_hover=False)
    dfv = dfv.loc[firsttime:lasttime]
    #PlotChargingProfile(dfvehicle=dfv, id=id, plot_efficiency_and_SOCmin=False)
    DFVtest.append(dfv)

# Export list of vehicles
with open('data/MPC-ready/df_TEST_vehicle_list.pkl', 'wb') as f:
    pickle.dump(DFVtest, f)

