"""
    Implementation of the economic MPC problem for multi-day Smart Charging of EVs.
    Now including price predictions - obtaind from https://github.com/solmoller/Spotprisprognose
    Spot price data is obtaind by running data_read.ipynb
    Prediction data is extracted and put together by running > ./data_extracter.sh
    Prediction data is pre-proccessed and then exported by running > python3 prognosis_csv_read.py
    The problem is solved using the PuLP package.
"""
# Imports
from pulp import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt
from mpc1 import PerfectForesight, plot_EMPC, DumbCharge, DayAhead
2+2

# Read the dfp and dft
dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv', sep=',', header=0, parse_dates=True)

dft['Atime'] = pd.to_datetime(dft['Atime'], format='%Y-%m-%d %H:%M:%S')
dfp['Atime'] = pd.to_datetime(dfp['Atime'], format='%Y-%m-%d %H:%M:%S')



##### External variables (SIMULATED)
plugin = 17.25; plugout = 7.15;
# Parameters of the battery
battery_size = 60 # kWh
b0 = 0.8 * battery_size
bmax = 1 * battery_size
xmax = 7  # kW (max charging power)
    
# User input
bmin_morning = 0.40 * battery_size;

h = len(dfp.columns) -2 -1
diff = (dfp['Atime'].iloc[-1].floor('H') - dfp['Atime'].iloc[0].ceil('H'))
N = int(diff.days*24 + diff.seconds/3600 + h+1)
T = N # Length of experiment
tvec = np.arange(T)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
u = np.random.uniform(8,16,T) * (tvec % 24 == np.floor(plugin)-1)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])
########

# Horizon of multi-day
h = len(dfp.columns)-2 -1 # t=0..h

def MultiDay(dfp, u, z, h, b0, bmax, bmin, xmax, c_tilde):
    diff = (dfp['Atime'].iloc[-1].floor('H') - dfp['Atime'].iloc[0].ceil('H'))
    N = int(diff.days*24 + diff.seconds/3600 + h+1)
    L = N-h # Length of experiment
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan

    # Loop over all hours, where there is still h hours remaining of the data
    for i in range(0, L):
        if i%50 == 0: print("i = " + str(i) + " of " + str(L))
        
        # Subset input
        tvec_i = np.arange(i, i+h+1)
        c = df['DKK'][tvec_i].to_numpy()
        #c_tilde = np.quantile(c, 0.1) # 10 % quantile

        # Find relevant input at the specific hours of flexibility
        z_i = z[tvec_i]
        u_i = u[tvec_i]
        bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

        # Solve
        prob, x, b = PerfectForesight(b0, bmax, bmin_i, xmax, c, c_tilde, u_i, z_i, h, tvec, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
 
        # Implement/store only the first step, and re-run in next hour
        x0 = value(x[0]); X[i]=x0;      # Amount charged in the now-hour
        b1 = value(b[1]); B[i+1]=b1;    # Battery level after the now-hour / beggining of next hour
        b0 = b1                         # Next SOC start is the current SOC

    # Costs
    total_cost = np.dot(X,df['DKK'][0:L]) - c_tilde * (B[-1] - B[0])

    # Tie results intro prob
    prob = {'x':X, 'b':B, 'u':u[0:L], 'c':df['DKK'][0:L], 'objective':total_cost}
    return(prob, x, b)

# Run the problem
prob, x, b = MultiDay(df, u, z, h, b0, bmax, bmin, xmax, c_tilde)
plot_EMPC(prob, 'Multi-Day Smart Charge', export=True)
