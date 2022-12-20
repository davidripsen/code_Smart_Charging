"""
Collection of functions for Smart Chargng
"""
from pulp import *
import numpy as np
import plotly.graph_objects as go
import datetime
import pandas as pd

def PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, r=1, verbose=True):
    # Init problem 
    prob = LpProblem("mpc1", LpMinimize)

    # Init variabless
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
    b[0] = b0

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * ((b[T+1])-b[0]))

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t]*r - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t+1] <= bmax
        prob += x[t] <= xmax * z[t]
        prob += x[t] >= 0

    # Solve problem
    if verbose:
        prob.solve(PULP_CBC_CMD(msg=1))
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Return results
    return(prob, x, b)

def ImperfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u_t_true, u_forecast, z, T, tvec, r, verbose=False):
    # Init problem 
    prob = LpProblem("mpc1", LpMinimize)

    # Init variabless
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax*1.25, cat='Continuous')
    s = LpVariable.dicts("s", tvec, lowBound=0, upBound=0.25*bmax, cat='Continuous') # Add penalizing slack for violating bmax=80%, but still remain below 100%
    #s2 = LpVariable.dicts("s2", np.append(tvec,T+1), lowBound=0, upBound=list(bmin), cat='Continuous') # Add penalizing slack for violating bmin, but still remain above 0%
    s2 = {i: LpVariable("s2_"+str(i), lowBound=0, upBound=ub) for i, ub in enumerate(bmin)}
    b[0] = b0; #s2[0] = 0;

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * ((b[T+1])-b[0]) + [c_tilde*100*s[t] + c_tilde*100*s2[t+1] for t in tvec])

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t]*r - u_forecast[t]
        prob += b[t+1] >= bmin[t+1] - s2[t+1] # Punishment slack variable for violating bmin at t+1
        prob += b[t+1] <= bmax + s[t]  # Punishment slack variable for violating bmax at t
        prob += x[t] <=   xmax * z[t]
        prob += x[t] >= 0

    # Solve problem
    if verbose:
        prob.solve(PULP_CBC_CMD(msg=1))
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Update b1 with actual use (relative to what we chose to charge)
    b[1] = b[0] + x[0]*r - u_t_true
    prob.assignVarsVals({'b_1': b[1]})

    # Return results
    return(prob, x, b)


def plot_EMPC(prob, name="", x=np.nan, b=np.nan, u=np.nan, c=np.nan, z=np.nan, starttime='', endtime='', export=False, BatteryCap=60, firsthour=0):
        # Identify iterative-appended, self-made prob
    fig = go.Figure()
    if type(prob) == dict:
        tvec = np.arange(0, len(prob['x']))
        tvec_b = np.arange(0, len(prob['b']))
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(prob['b'][t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['u'][t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['x'][t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['c'][t]) for t in tvec], mode='lines', name='True Price'))
        fig.add_trace(go.Scatter(x=tvec, y=[prob['z'][t]*2-1 for t in tvec], mode='lines', name='Plugged-in', line=dict(color='black', width=0.5)))
        obj = prob['objective']
    else:
        tvec = np.arange(0, len(x))
        tvec_b = np.arange(0, len(b))
        obj = value(prob.objective)
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(b[t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='True Price'))
        fig.add_trace(go.Scatter(x=tvec, y=[z[t]*2-1 for t in tvec], mode='lines', name='Plugged-in',  line=dict(color='black', width=0.5)))
    
    fig.update_xaxes(tickvals=tvec_b[::24]+firsthour, ticktext=[str(t//24) for t in tvec_b[::24]+firsthour])

    # Fix y-axis to lie between 0 and 65
    fig.update_yaxes(range=[-3, BatteryCap+2])

    # add "Days" to x-axis
    # Add total cost to title
    fig.update_layout(title=name + "    from " + starttime +" to "+ endtime+"      Total cost: " + str(round(obj)) + " DKK  (+tariffs)",
        xaxis_title="Days",
        yaxis_title="kWh  or  DKK/kWh  or  Plugged-in [T/F]")
    fig.show()

    ## Export figure
    if export:
        fig.write_html( "plots/MPC/" + name + "_mpc.html")

def DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, r=1, verbose=False):
    # Init problem
    prob = LpProblem("mpc_DumbCharge", LpMinimize)

    # Init variables
    global x
    global b
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
    i = LpVariable.dicts("i", tvec, lowBound=0, upBound=1, cat='Binary')
    s = LpVariable.dicts("s", tvec, lowBound=0, upBound=0.20*1.25*bmax, cat='Continuous')
    #s2 = LpVariable.dicts("s2", tvec, lowBound=0, upBound=bmin, cat='Continuous')
    s2 = {i: LpVariable("s2_"+str(i), lowBound=0, upBound=ub, cat='Continuous') for i, ub in enumerate(bmin)}
    b[0] = b0
    M = 10**6

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]) + [100*c_tilde*(s[t]+s2[t+1]) for t in tvec])

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t]*r - u[t]
        prob += b[t+1] >= bmin[t+1] - s2[t+1]
        prob += b[t+1] <= bmax + s[t]
        
        ######## DUMB CHARGE ########
        ########## Implement in OR-terms: x[t] == min(z[t]*xmax, bmax-b[t]) ==> min(z[t]*xmax, bmax+s[t]-b[t]-u[t] / r)
        # Ensure i[t] == 1, if z[t]*xmax < bmax-b[t] (dvs. i=1 når der er rigeligt plads på batteriet)
        prob += (bmax+s[t]-b[t])/r - z[t]*xmax  - M*i[t] <= 0    # Sry, men tilføj evt. u[t], fordi der i analysen bliver forbrugt strøm mens vi oplader. I praksis ville denne være 0 (eller næsten 0) og kan slettes fra modellen. Hvorfor er det ikke snyd at have den med: I Dumbcharge vil vi alligevel oplade med max hastighed indtil den er fuld, så hvis der lineært blet forbrugt strøm over den time, er det effektivt det samme at kende u[t] i den time.
        prob += z[t]*xmax - (bmax+s[t]-b[t])/r - M*(1-i[t]) <= 0

        # Use i[t] to constraint x[t]
        prob += x[t] <= z[t]*xmax
        prob += x[t] <= (bmax+s[t]-b[t]) / r             # s[t] er tilføjet for at undgå, at x[t] tvinges negativ, når b[t] er en smule højere end bmax
        prob += x[t] >= (z[t]*xmax - M*(1-i[t]))         # i = 1 betyder, at der lades max kapacitet
        prob += x[t] >= (bmax+s[t]-b[t] - M*i[t])  / r   # i = 0 betyder, at vi kun kan lade de resterende til 80 eller 100 % SOC
        #prob += i[t] <= z[t]

    # Solve problem
    prob.solve(PULP_CBC_CMD(gapAbs = 0.01, msg=verbose))

    # Return objective without penalization
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]))

    # Return results
    return(prob, x, b)



def ExtractEVdataForMPC(dfv, z_var, u_var, uhat_var, bmin_var, p):
    # Read the dfp and dft and dfspot --- This section can be moved out of the function to save a slgiht bit of time
    dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
    dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
    dfspot = pd.read_csv('data/spotprice/df_spot_commontime.csv', sep=',', header=0, parse_dates=True)

    dft['Atime'] = pd.to_datetime(dft['Atime'], format='%Y-%m-%d %H:%M:%S')
    dfp['Atime'] = pd.to_datetime(dfp['Atime'], format='%Y-%m-%d %H:%M:%S')
    dfspot['Time'] = pd.to_datetime(dfspot['Time'], format='%Y-%m-%d %H:%M:%S')

    # Convert timezone from UTC to Europe/Copenhagen
    dfspot['Time'] = dfspot['Time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
    dfp['Atime'] = dfp['Atime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
    dft['Atime'] = dft['Atime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')

    ####################### Load each element in the list into a dataframe ############################
    starttime = max(dfspot['Time'][0], dfp['Atime'][0], dfv.index[0])
    endtime = min(dfspot['Time'].iloc[-1], dfp['Atime'].iloc[-1], dfv.index[-1])

    # Cut dfs to be withing starttime and endtime
    dfspot = dfspot[(dfspot['Time'] >= starttime) & (dfspot['Time'] <= endtime)].reset_index(drop=True)
    #dfp = dfp[(dfp['Atime'] >= starttime) & (dfp['Atime'] <= endtime)].reset_index(drop=True) # The forecast history is the bottleneck
    #dft = dft[(dft['Atime'] >= starttime) & (dft['Atime'] <= endtime)].reset_index(drop=True)
    dfv = dfv[(dfv.index >= starttime) & (dfv.index <= endtime)]
    timestamps = dfv.index
    firsthour = dfv.index[0].hour
    dfp = dfp[(dfp['Atime'] >= timestamps[0]) & (dfp['Atime'] <= timestamps[-1])].reset_index(drop=True) # The forecast history is the bottleneck
    dft = dft[(dft['Atime'] >= timestamps[0]) & (dft['Atime'] <= timestamps[-1])].reset_index(drop=True)
    dfv = dfv.reset_index(drop=True)

    ## Print occurences of number of hours between forecasts
    #dfp.Atime_diff.value_counts() # Up to 66 hours between forecasts

    ############################################ EXTRACT EV USAGE DATA ####################################################
    # Use
    vehicle_id = dfv['vehicle_id'].unique()[0]
    z = ((dfv[z_var] == 1)*1).to_numpy()
    u = dfv[u_var].to_numpy()
    uhat = dfv[uhat_var].to_numpy()
    uhat = np.append(0, uhat) # For first iter, uhat = 0 => uhat[k] = RollingMean(use)_{i = k-(10*24)...k-1}
    b0 = dfv['SOC'][0]
    r = dfv['efficiency_median'].unique()[0]
    #print(np.sum(u), "==", np.sum(dfv['use']))
    # Input
    bmin = dfv[bmin_var].to_numpy()
    # Vehicle parameters
    bmax = dfv['SOCmax'].median()
    #bmax = np.nanmin([dfv['SOCmax'], dfv['BatteryCapacity']], axis=0)
    xmax = dfv['CableCapacity'].unique()[0]
    c_tilde = np.quantile(dfspot['TruePrice'], p) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

    return dfv, dfspot, dfp, dft, timestamps, z, u, uhat, b0, r, bmin, bmax, xmax, c_tilde, vehicle_id, firsthour, starttime, endtime

def PlotChargingProfile(D2=None, dfvehicle=None, var="VEHICLE_ID", id=13267, plot_efficiency_and_SOCmin=True, vertical_hover=False, df_only=False):
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

            # Extract charge from 'KWHS' and add to df where time is the same
            xt = pd.DataFrame(eval(D2v.iloc[i]['KWHS']))
            if D2v.iloc[i]['KWH'] != round(xt.sum()[1],4):
                print("KWH total and sum(kWh_t) does not match for D2v row i=", i)
            xt['time'] = pd.to_datetime(xt['time'])
            xt['time'] = xt['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
            xt = xt.set_index('time')
            df.loc[xt.index, 'charge'] = xt['value']

            # Efficiency of charging (ratio of what has been charged to what goes into the battery)
            #if D2v.iloc[i]['KWH'] >= 1: # Only proper charging
            if D2v.iloc[i]['KWH'] > 0:
                df.loc[D2v.iloc[i]['CABLE_PLUGGED_IN_AT']:D2v.iloc[i]['RELEASED_AT'], 'efficiency'] = ((D2v.iloc[i].SOC - D2v.iloc[i].SOC_START) / 100 * D2v.iloc[i]['capacity_kwh']) / D2v.iloc[i].KWH

            # Add the right spot prices to df
            if type(D2v.iloc[i]['SPOT_PRICES']) == str and len(eval(D2v.iloc[i]['SPOT_PRICES'])) != 0:
                prices = pd.DataFrame(eval(D2v.iloc[i]['SPOT_PRICES']))
                prices['time'] = pd.to_datetime(prices['time'])
                prices['time'] = prices['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Copenhagen')
                prices = prices.set_index('time')
                df.loc[prices.index, 'price'] = prices['value']
            
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
        df['z_plan_everynight'] = df['z_plan']
        df.loc[(df.index.hour >= 22) | (df.index.hour < 6), 'z_plan_everynight'] = 1

        # bmin_everymorning:
        df['SOCmin_everymorning'] = df['SOCmin']
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
        df['efficiency_median'] = df['efficiency'].median()

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
    name = "Plugged-in (actual) [true/false]",
    line=dict(
        color='black',
        dash='dot',
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
        y=df['z_plan_everynight'],
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
            color='green',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['use'],
        mode='lines',
        name='Use [kWh]',
        line=dict(
            color='red',
            width=2
        )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['use_lin'],
        mode='lines',
        name='Use (from interpolated SOC) [kWh]',
        line=dict(
            color='red',
            width=2,
            dash='dot'
        )
    ))

    fig.add_trace(go.Scatter(
    x=df.index,
    y=df['use_rolling'],
    mode='lines',
    name='Use ('+str(7)+' day rolling mean) [kWh]',
    line=dict(
        color='red',
        width=2,
        dash='dot'
    )
    ))

    fig.add_trace(go.Scatter(
    x=df.index,
    y=df['use_ewm'],
    mode='lines',
    name='Use (Exponentially Weighted Moving Average with half life = '+str(2)+') [kWh]',
    line=dict(
        color='red',
        width=2,
        dash='dash'
    )
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['use_dailyaverage'],
        mode='lines',
        name='Use daily average (outside of plug-in) [kWh]',
        line=dict(
            color='red',
            width=0.5,
            dash='dash'
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
    y=df['trueprice'],
    mode='lines',
    name='Price (EnergiDataService.dk) [DKK/kWh excl. tarifs]',
    line=dict(
        color='purple',
        width=1,
        dash='dash'
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
            name = "Input: Minimum SOC (assumption)",
            line=dict(
                color='lightblue',
                width=2,
                dash='dash'
            )
            ))

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
            color='lightblue',
            width=2
        )
    ))

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
        yaxis_title_text="kWh or DKK/kWh", # yaxis label
        #font=dict(
        #    size=18,
        #    color="RebeccaPurple"
        #)
    )
    if not df_only:
        fig.show()
    return df