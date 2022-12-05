"""
Implementation of the economic MPC problem for multi-day Smart Charging of EVs on data from Monta.
"""

from pulp import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt
import os
from code_Smart_Charging.MPC.FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge
runMany = True


# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Load each element in the list into a dataframe
dfv = DFV[3]  #dfv1, dfv2, dfv3, dfv4, dfv5, dfv6, dfv7, dfv8, dfv9 = DFV[1], DFV[2], DFV[3], DFV[4], DFV[5], DFV[6], DFV[7], DFV[8], DFV[9]
    # Is dfv2 broke?

# Read the dfp and dft and dfspot
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

starttime = max(dfspot['Time'][0], dfp['Atime'][0], dfv.index[0])
endtime = min(dfspot['Time'].iloc[-1], dfp['Atime'].iloc[-1], dfv.index[-1])

# Cut dfs to be withing starttime and endtime
dfspot = dfspot[(dfspot['Time'] >= starttime) & (dfspot['Time'] <= endtime)].reset_index(drop=True)
dfp = dfp[(dfp['Atime'] >= starttime) & (dfp['Atime'] <= endtime)].reset_index(drop=True)
dft = dft[(dft['Atime'] >= starttime) & (dft['Atime'] <= endtime)].reset_index(drop=True)
dfv = dfv[(dfv.index >= starttime) & (dfv.index <= endtime)].reset_index(drop=True)


# Print occurences of number of hours between forecasts
dfp.Atime_diff.value_counts() # Up to 66 hours between forecasts

############################################ EXTRACT EV USAGE DATA ####################################################

# Choice of fit for use
u_fit = 'use_lin' # 'use_ewm': Exponential weighted moving average
z_var = 'z_plan' # 'z_plan': All historical plug-ins (and planned plug-out).  'z_plan_everynight': z_plan + plug-in every night from 22:00 to 06:00

#### Extract EV usage from Monta data #######
# Use
vehicle_id = dfv['vehicle_id'].unique()[0]
z = ((dfv[z_var] == 1)*1).to_numpy()
u = dfv[u_fit].to_numpy()
uhat = dfv['use_rolling'].to_numpy()
b0 = dfv['SOC'][0]
r = dfv['efficiency_median'].unique()[0]
# Input
bmin = dfv['SOCmin'].to_numpy()
# Vehicle parameters
bmax = dfv['SOCmax'].median()
#bmax = np.nanmin([dfv['SOCmax'], dfv['BatteryCapacity']], axis=0)
xmax = dfv['CableCapacity'].unique()[0]
# Price
c_tilde = np.quantile(dfspot['TruePrice'], 0.1) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

# Horizon
h=5*24


#################################################### LET'S GO! ########################################################


#### Tasks:
# Modify function such that bmax can be a series, not just a scalar
def MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh=5*24):
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
                          # maxh = maximum h of interest ==> to allow comparison on exact same data for different horizons h.

    # Init
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan
    c = dfspot['TruePrice'].to_numpy()
    costs = 0
    k = 0
    
    
    # For each Atime
    for i in range(len(dfp)):
        # For each hour until next forecast
        for j in range(dfp['Atime_diff'][i]):
            if k%50 == 0: print("k = " + str(k) + " of " + str(L-1))
            # Extract forecasts from t=0..h
            c_forecast = dfp.iloc[i, (j+2):(j+2+h+1)].to_numpy() 
            tvec_i = np.arange(k, k+h+1)

            # Find relevant input at the specific hours of flexibility
            z_i = z[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1) #u_i = u[tvec_i]
            u_t_true = u[k]

            # Solve
            prob, x, b = ImperfectForesight(b0, bmax, bmin_i, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z_i, h, tvec, r, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
    
            # Implement/store only the first step, and re-run in next hour
            x0 = value(x[0]); X[k]=x0;                # Amount charged in the now-hour
            b1 = value(b[1]); B[k+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * c[k];     # Cost of charging in the now-hour
            b0 = b1                                   # Next SOC start is the current SOC
            k += 1

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'objective':total_cost}
                return(prob, x, b)

### Run the problem
if not runMany:
    h = 5*24 # 5 days horizon for the multi-day smart charge
    prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r)
    plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)




#################################################### RUN ALL THE MODELS ########################################################




### Perfect Foresight
    # Compare models on the data within horizon
L = len(u) - (h+1)
T = L - 1
tvec = np.arange(T+1)
maxh = 5*24
T_within = T #-maxh 
c_within = dfspot['TruePrice'][0:T_within+1] # Actually uses all prices in this case:-)
tvec_within = tvec[0:T_within+1]
z_within = z[0:T_within+1]
u_within = u[0:T_within+1]
bmin_within = bmin[0:T_within+2]

### Perfect Foresight
prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r, verbose=True)
plot_EMPC(prob, 'Perfect Foresight   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within,  starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)

### Multi-Day SmartCharge
if runMany:
    for h in range(1,6):
        print("h = " + str(h))
        prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r)
        plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(h)+' days) of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax)
        print("Total cost: " + str(prob['objective']))
        print("")

### DumbCharge
prob, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within)
if LpStatus[prob.status] == 'Optimal':
    plot_EMPC(prob, 'Dumb Charge   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within, starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)
else:
    print("DumbCharge failed on this set of simulated data")