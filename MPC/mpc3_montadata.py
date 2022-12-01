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
# Load local functions
os.chdir('/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/code_Smart_Charging/MPC')
from FunctionCollection import PerfectForesight, plot_EMPC, DumbCharge
os.chdir('/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge')
runMany = True


# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Load each element in the list into a dataframe
dfv = DFV[8]  #dfv1, dfv2, dfv3, dfv4, dfv5, dfv6, dfv7, dfv8, dfv9 = DFV[1], DFV[2], DFV[3], DFV[4], DFV[5], DFV[6], DFV[7], DFV[8], DFV[9]
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

# Choice of fit for use
u_fit = 'use_ewm' # 'use_ewm': Exponential weighted moving average

#### Extract EV usage from Monta data #######
# Use
vehicle_id = dfv['vehicle_id'].unique()[0]
z = ((dfv['z_plan'] == 1)*1).to_numpy()
u = dfv[u_fit].to_numpy()
b0 = dfv['SOC'][0]
# Input
bmin = dfv['SOCmin'].to_numpy()
# Vehicle parameters
bmax = dfv['SOCmax'].median()
#bmax = pd.Series(np.nanmin([dfv['SOCmax'], dfv['BatteryCapacity']], axis=0), index=dfv.index).to_numpy()
xmax = dfv['CableCapacity'].unique()[0]
# Price
c_tilde = np.quantile(dfspot['TruePrice'], 0.1) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

# Horizon
h=5*24








# ######### External variables from EV USE (SIMULATED)
# plugin = 17.25; plugout = 7.15;
# # Parameters of the battery
# battery_size = 60 # kWh
# b0 = 0.8 * battery_size
# bmax = 1 * battery_size
# xmax = 7  # kW (max charging power)

# # User input
# bmin_morning = 0.40 * battery_size;

# # Horizon (!)
# h = 5*24 # 5 days horizon for the multi-day smart charge


# # External variables (SIMULATED) - delete upon recieving true user data
# diff = (dfspot['Time'].iloc[-1].floor('H') - dfp['Atime'].iloc[0].ceil('H'))
# T = int(diff.days*24 + diff.seconds/3600) +h
# tvec = np.arange(T+1)
# z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
# u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1) # uniform(8,16, T eller T+1? MANGLER)
# bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])

#### Tasks:
# Modify function such that bmax can be a series, not just a scalar
# Define true L(ength) = 1190 - (h+1)
# Does z, u, bmin correspond to the i-loop times? 
def MultiDay(dfp, dfspot, u, z, h, b0, bmax, bmin, xmax, c_tilde, maxh=5*24):
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
                          # maxh = maximum h of interest ==> to allow comparison on exact same data for different horizons h.

    # Init
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan
    costs = 0
    k = 0
    
    # For each Atime
    for i in range(len(dfp)):
        # For each hour until next forecast 
        for j in range(dfp['Atime_diff'][i]):
            if k%50 == 0: print("k = " + str(k) + " of " + str(L-1))
            # Extract forecasts from t=0..h
            c = dfp.iloc[i, (j+2):(j+2+h+1)].to_numpy() 
            tvec_i = np.arange(k, k+h+1)

            # Find relevant input at the specific hours of flexibility
            z_i = z[tvec_i]
            u_i = u[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            # Solve
            prob, x, b = PerfectForesight(b0, bmax, bmin_i, xmax, c, c_tilde, u_i, z_i, h, tvec, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
    
            # Implement/store only the first step, and re-run in next hour
            x0 = value(x[0]); X[k]=x0;                # Amount charged in the now-hour
            b1 = value(b[1]); B[k+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * dfspot['TruePrice'][k];     # Cost of charging in the now-hour
            b0 = b1                                   # Next SOC start is the current SOC
            k += 1

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':dfspot['TruePrice'][0:L], 'objective':total_cost}
                return(prob, x, b)

### Run the problem
if not runMany:
    h = 5*24 # 5 days horizon for the multi-day smart charge
    prob, x, b = MultiDay(dfp, dfspot, u, z, h, b0, bmax, bmin, xmax, c_tilde)
    plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)












# if runMany:
#     for h in range(1,6):
#         print("h = " + str(h))
#         prob, x, b = MultiDay(dfp, dfspot, u, z, h*24, b0, bmax, bmin, xmax, c_tilde)
#         plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(h)+' days)', starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax)
#         print("Total cost: " + str(prob['objective']))
#         print("")

#### DumbCharge
#h2=0; c_tilde0 = 0;
#prob, x, b = MultiDay(dfp, dfspot, u, z, h2, b0, bmax, bmin, xmax, c_tilde0)
#plot_EMPC(prob, 'Dumb Charge', starttime=starttime, endtime=endtime, export=False)

#### Day-Ahead SmartCharge
#h3=36; c_tilde0=0;
#prob, x, b = MultiDay(dfk, dfspot, u, z, h3, b0, bmax, bmin, xmax, c_tilde0)
#plot_EMPC(prob, 'Day-Ahead Smart Charge', starttime=starttime, endtime=endtime, export=False)

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
prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, verbose=True)
plot_EMPC(prob, 'Perfect Foresight   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within,  starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)

### Day-Ahead SmartCharge
if runMany:
    for h in range(1,6):
        print("h = " + str(h))
        prob, x, b = MultiDay(dfp, dfspot, u, z, h*24, b0, bmax, bmin, xmax, c_tilde)
        plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(h)+' days) of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax)
        print("Total cost: " + str(prob['objective']))
        print("")

### DumbCharge
prob, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within)
if LpStatus[prob.status] == 'Optimal':
    plot_EMPC(prob, 'Dumb Charge   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within, starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax)
else:
    print("DumbCharge failed on this set of simulated data")