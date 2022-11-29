"""
Implementation of the economic MPC problem for multi-day Smart Charging of EVs on data from Monta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime

# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Load each element in the list into a dataframe
dfv = DFV[0]  #dfv1, dfv2, dfv3, dfv4, dfv5, dfv6, dfv7, dfv8, dfv9 = DFV[1], DFV[2], DFV[3], DFV[4], DFV[5], DFV[6], DFV[7], DFV[8], DFV[9]

# Read the dfp and dft and dfspot
dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
dfspot = pd.read_csv('data/spotprice/df_spot_commontime.csv', sep=',', header=0, parse_dates=True)
trueprice = dfspot['TruePrice'].to_numpy()

dft['Atime'] = pd.to_datetime(dft['Atime'], format='%Y-%m-%d %H:%M:%S')
dfp['Atime'] = pd.to_datetime(dfp['Atime'], format='%Y-%m-%d %H:%M:%S')
dfspot['Time'] = pd.to_datetime(dfspot['Time'], format='%Y-%m-%d %H:%M:%S')
starttime = str(dfspot['Time'][0].date())
endtime = str(dfspot['Time'].iloc[-1].date())






















######### External variables from EV USE (SIMULATED)
plugin = 17.25; plugout = 7.15;
# Parameters of the battery
battery_size = 60 # kWh
b0 = 0.8 * battery_size
bmax = 1 * battery_size
xmax = 7  # kW (max charging power)

# User input
bmin_morning = 0.40 * battery_size;

# Horizon (!)
h = 5*24 # 5 days horizon for the multi-day smart charge
c_tilde = np.quantile(trueprice, 0.1) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

# External variables (SIMULATED) - delete upon recieving true user data
diff = (dfspot['Time'].iloc[-1].floor('H') - dfp['Atime'].iloc[0].ceil('H'))
T = int(diff.days*24 + diff.seconds/3600) +h
tvec = np.arange(T+1)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1) # uniform(8,16, T eller T+1? MANGLER)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])


def MultiDay(dfp, dfspot, u, z, h, b0, bmax, bmin, xmax, c_tilde):
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    diff = (dfspot['Time'].iloc[-1].floor('H') - dfp['Atime'].iloc[0].ceil('H'))
    L = int(diff.days*24 + diff.seconds/3600) +1

    # Init
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan
    costs = 0
    cnt = 0
    
    # For each Atime
    for i in range(len(dfp)):
        if i%5 == 0: print("i = " + str(i) + " of " + str(len(dfp)))
        # For each hour until next forecast 
        for j in range(dfp['Atime_diff'][i]):
            # Extract forecasts from t=0..h
            c = dfp.iloc[i, (j+2):(j+2+h+1)].to_numpy() 
            tvec_i = np.arange(cnt, cnt+h+1)

            # Find relevant input at the specific hours of flexibility
            z_i = z[tvec_i]
            u_i = u[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            # Solve
            prob, x, b = PerfectForesight(b0, bmax, bmin_i, xmax, c, c_tilde, u_i, z_i, h, tvec, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
    
            # Implement/store only the first step, and re-run in next hour
            x0 = value(x[0]); X[cnt]=x0;                # Amount charged in the now-hour
            b1 = value(b[1]); B[cnt+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * dfspot['TruePrice'][cnt];     # Cost of charging in the now-hour
            b0 = b1                                     # Next SOC start is the current SOC
            cnt += 1

    # Costs
    total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

    # Tie results intro prob
    prob = {'x':X, 'b':B, 'u':u[0:L], 'c':dfspot['TruePrice'][0:L], 'objective':total_cost}
    return(prob, x, b)

### Run the problem
h = 5*24 # 5 days horizon for the multi-day smart charge
prob, x, b = MultiDay(dfp, dfspot, u, z, h, b0, bmax, bmin, xmax, c_tilde)
plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(int(h/24))+'days)', starttime=starttime, endtime=endtime, export=True)

if runMany:
    for h in range(1,6):
        print("h = " + str(h))
        prob, x, b = MultiDay(dfp, dfspot, u, z, h*24, b0, bmax, bmin, xmax, c_tilde)
        plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(h)+' days)', starttime=starttime, endtime=endtime, export=True)
        print("Total cost: " + str(prob['objective']))
        print("")

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
T_within = T - h
c_within = dfspot['TruePrice'][0:T_within+1] # Actually uses all prices in this case:-)
tvec_within = tvec[0:T_within+1]
z_within = z[0:T_within+1]
u_within = u[0:T_within+1]
bmin_within = bmin[0:T_within+2]
prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, verbose=True)
plot_EMPC(prob, 'Perfect Foresight', x, b, u_within, c_within, starttime=starttime, endtime=endtime, export=False)

### DumbCharge
prob, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within)
if LpStatus[prob.status] == 'Optimal':
    plot_EMPC(prob, 'Dumb Charge', x, b, u_within, c_within, starttime=starttime, endtime=endtime, export=False)
else:
    print("DumbCharge failed on this set of simulated data")