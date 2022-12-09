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
from code_Smart_Charging.MPC.FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge
runMany = True

# Read scenarios from txt
scenarios = np.loadtxt('./data/MPC-ready/scenarios.csv', delimiter=','); scenarios_all=scenarios;

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


# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

####################### Load each element in the list into a dataframe
dfv = DFV[0]  #dfv1, dfv2, dfv3, dfv4, dfv5, dfv6, dfv7, dfv8, dfv9 = DFV[1], DFV[2], DFV[3], DFV[4], DFV[5], DFV[6], DFV[7], DFV[8], DFV[9]
dfv              # Is DFV[3] broke?

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
# Choice of variables to use
u_var = 'use_lin' # 'use_lin' or 'use'
z_var = 'z_plan_everynight' # 'z_plan': All historical plug-ins (and planned plug-out).  'z_plan_everynight': z_plan + plug-in every night from 22:00 to 06:00
bmin_var = 'SOCmin_everymorning' # SOCmin <=> min 40 % hver hver egentlige plugout. SOCmin_everymorning <=> også min 40 % hver morgen.

#### Extract EV usage from Monta data #######
# Use
vehicle_id = dfv['vehicle_id'].unique()[0]
z = ((dfv[z_var] == 1)*1).to_numpy()
u = dfv[u_var].to_numpy()
uhat = dfv['use_lin'].to_numpy()
b0 = dfv['SOC'][0]
r = dfv['efficiency_median'].unique()[0]
# Input
bmin = dfv[bmin_var].to_numpy()
# Vehicle parameters
bmax = dfv['SOCmax'].median()
#bmax = np.nanmin([dfv['SOCmax'], dfv['BatteryCapacity']], axis=0)
xmax = dfv['CableCapacity'].unique()[0]
# Price
c_tilde = np.quantile(dfspot['TruePrice'], 0.2) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h




#################################################### LET'S GO! ########################################################


#################################################### RUN ALL THE MODELS ########################################################




# Note: Scenarios as is right now, do not take into account that the uncertainty/scenarios differences are very dependent of time of day.

def StochasticProgram(scenarios, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, verbose=True):
    """
    Solves the 2-stage stochastic program for a given time horizon T, and returns the optimal solution.
    l: Length of deterministic prices
    O: Number of scenarios (Omega)
    """
    scenarios = scenarios_all[77:77+10, :]; O=len(scenarios); # for Dev: Antag 2 scenarier
    #scenarios = scenarios_all
    O, K = scenarios.shape
    tvec_d = tvec[0:l] # Deterministic part
    tvec_s = tvec[l:]  # Stochastic part
    c_d = c_forecast[:l] # Deterministic part
    c_s = c_forecast + scenarios # Stochastic part

    # Init problem
    prob = LpProblem("StochEcoMPC", LpMinimize)

    # Init variabless
    x_d = LpVariable.dicts("x_d", tvec_d, lowBound=0, upBound=xmax, cat='Continuous')
    x_s = LpVariable.dicts("x_s", [(t,o) for o in range(O) for t in tvec_s], lowBound=0, upBound=xmax, cat='Continuous') #xs_i,omega
    b = LpVariable.dicts("b", [(t,o) for o in range(O) for t in np.append(tvec,tvec[-1]+1)], lowBound=0, upBound=bmax, cat='Continuous')
    # Set initial SOC to b0 for all scenarios o
    for o in range(O): b[(0,o)] = b0

    # Objective
    prob += lpSum([c_d[t]*x_d[t] for t in tvec_d]) + lpSum([1/O * c_s[o,t]*x_s[t,o] for t in tvec_s for o in range(O)]) - lpSum([1/O * c_tilde * ((b[tvec[-1],o]) - b[0,o]) for o in range(O)])

    # Constraints
    for t in tvec_d: # Deterministic part
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_d[t]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1]
            prob += b[(t+1,o)] <= bmax
            prob += x_d[t] <= xmax * z[t]
            prob += x_d[t] >= 0

    for t in tvec_s: # Stochastic part
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_s[(t,o)]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1]
            prob += b[(t+1,o)] <= bmax
            prob += x_s[(t,o)] <= xmax * z[t]
            prob += x_s[(t,o)] >= 0

    # Solve problem
    if verbose:
        prob.solve()
    else:
        prob.solve(PULP_CBC_CMD(msg=0))
        print("Status:", LpStatus[prob.status])

    #Update b1 with actual use (relative to what we chose to charge) (Should be sufficient only to update b(1,0))
    for o in range(O):
        b[1,o] = b[0,o] + value(x_d[0]) - u_t_true
        prob.assignVarsVals({'b_(1,_'+str(o)+')': b[1,o]})
        assert b[1,o] == value(b[1,0]), "b(1,o) is not equal to value(b(1,0))"
        # ^ Most of this code is redundant

    # Return results
    return(prob, x_d, b, x_s)

prob, x_d, b, x_s = StochasticProgram(scenarios, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, verbose=True)

# Print results nicely
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Plot the two scenarios
plt.figure()
plt.plot(c + scenarios[0,:], label='Scenario 0')
plt.plot(c + scenarios[1,:], label='Scenario 1')
plt.show()







#### Tasks:
# Modify function such that bmax can be a series, not just a scalar
def MultiDayStochastic(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, discountfactor=None, maxh=6*24):
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
            if k%50 == 1: print("k = " + str(k) + " of " + str(L-1))
            # Extract forecasts from t=0..h
            c_forecast = dfp.iloc[i, (j+2):(j+2+h+1)].to_numpy();
            tvec_i = np.arange(k, k+h+1)

            # Find relevant input at the specific hours of flexibility
            z_i = z[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1)
            #u_forecast = u[tvec_i]  # Snyd: Antag kendt Use
            u_t_true = u[k]

            l = 12 # slet

            # Solve
            prob, x_d, b, x_s = StochasticProgram(scenarios, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, verbose=False)

            # Implement/store only the first step, and re-run in next hour
            x0 = value(x_d[0]); X[k]=x0;                # Amount charged in the now-hour
            b1 = value(b[1,0]); B[k+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * c[k];     # Cost of charging in the now-hour
            b0 = b1                                   # Next SOC start is the current SOC
            k += 1

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'z':z[0:L], 'objective':total_cost}
                return(prob, x, b)

### Run the problem
h = 6*24 # 5 days horizon for the multi-day smart charge
prob, x, b = MultiDayStochastic(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24)
plot_EMPC(prob, 'Stochastic Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
