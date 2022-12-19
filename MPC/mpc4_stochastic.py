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
pd.set_option('display.max_rows', 500)
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
dfv = DFV[3]  #dfv1, dfv2, dfv3, dfv4, dfv5, dfv6, dfv7, dfv8, dfv9 = DFV[1], DFV[2], DFV[3], DFV[4], DFV[5], DFV[6], DFV[7], DFV[8], DFV[9]
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
c_tilde = np.quantile(dfspot['TruePrice'], 0.1) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h




#################################################### LET'S GO! ########################################################


#################################################### RUN ALL THE MODELS ########################################################




# Note: Scenarios as is right now, do not take into account that the uncertainty/scenarios differences are very dependent of time of day.
# Sample 5 integers between 0 and 99
n_scenarios=5
def StochasticProgram(scenarios, n_scenarios, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, verbose=True):
    """
    Solves the 2-stage stochastic program for a given time horizon T, and returns the optimal solution.
    l: Length of deterministic prices
    O: Number of scenarios (Omega)
    """
    scenarios = scenarios_all[33:33+n_scenarios, :] # for Dev: Antag n_scenarier scenarier
    #scenarios = scenarios_all
    O, K = scenarios.shape
    tvec_d = tvec[0:l] # Deterministic part
    tvec_s = tvec[l:]  # Stochastic part
    c_d = c_forecast[:l] # Deterministic part
    c_s = c_forecast + scenarios # Stochastic part
    c_s[c_s < 0] = 0 # Truncate cost_stochastic to assume non-negatice electricity spot prices
 
    # Init problem
    prob = LpProblem("StochEcoMPC", LpMinimize)

    # Init variabless
    x_d = LpVariable.dicts("x_d", tvec_d, lowBound=0, upBound=xmax, cat='Continuous')
    x_s = LpVariable.dicts("x_s", [(t,o) for o in range(O) for t in tvec_s], lowBound=0, upBound=xmax, cat='Continuous') #xs_i,omega
    b = LpVariable.dicts("b", [(t,o) for o in range(O) for t in np.append(tvec,tvec[-1]+1)], lowBound=0, upBound=bmax, cat='Continuous')
    s = LpVariable.dicts("s", [(t,o) for o in range(O) for t in tvec], lowBound=0, upBound=0.25*bmax, cat='Continuous') # Add penalizing slack for violating bmax=80%, but still remain below 100%
    s2 = {(i, o): LpVariable("s2_("+str(i)+",_"+str(o)+")", lowBound=0, upBound=ub) for o in range(O) for i, ub in enumerate(bmin)}
    # Set initial SOC to b0 for all scenarios o
    for o in range(O): b[(0,o)] = b0

    # Objective
    prob += lpSum([c_d[t]*x_d[t] for t in tvec_d]) + lpSum([1/O * c_s[o,t]*x_s[t,o] for t in tvec_s for o in range(O)]) - lpSum([1/O * c_tilde * ((b[tvec[-1],o]) - b[0,o]) for o in range(O)] + lpSum([1/O * 100*c_tilde*(s[t,o]+s2[t+1,o]) for t in tvec for o in range(O)]))

    # Constraints
    for t in tvec_d: # Deterministic part
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_d[t]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1] - s2[t+1,o]
            prob += b[(t+1,o)] <= bmax + s[t,o]
            prob += x_d[t] <= xmax * z[t]
            prob += x_d[t] >= 0

    for t in tvec_s: # Stochastic part
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_s[(t,o)]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1] - s2[t+1,o]
            prob += b[(t+1,o)] <= bmax + s[t,o]
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
        b[(1,o)] = b[(0,o)] + value(x_d[0]) - u_t_true
        prob.assignVarsVals({'b_(1,_'+str(o)+')': b[1,o]})
        assert b[1,o] == value(b[1,0]), "b(1,o) is not equal to value(b(1,0))"
        # ^ Most of this code is redundant

    # Return results
    return(prob, x_d, b, x_s)

prob, x_d, b, x_s = StochasticProgram(scenarios, 10, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l=12, verbose=True)

# Print results nicely
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)


# Plot scenarios
plt.figure()
for j in range(0, n_scenarios):
    #plt.plot(c_forecast + scenarios[j,:], label='Scenario '+str(j))
    plt.plot(c_s[j,:], label='Scenario '+str(j))
c = dft.iloc[i, (j+3):(j+3+h+1)].to_numpy();
plt.plot(c, label='True Price', color='black', linewidth=3)
plt.ylim(-0.5, 7)
plt.plot(c, label='True Price')
plt.show()








#### Tasks:
# Modify function such that bmax can be a series, not just a scalar
def MultiDayStochastic(scenarios, n_scenarios, dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, discountfactor=None, maxh=6*24, perfectForesight=False):
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    maxh=6*24 # Delete
    h = 6*24 # Delete
    perfectForesight = True
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
                        # maxh = maximum h of interest ==> to allow comparison on exact same data for different horizons h.
    L = 500 # Delete

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
            c_forecast = dfp.iloc[i, (j+3):(j+3+h+1)].to_numpy();
            if perfectForesight:
                c_forecast = dft.iloc[i, (j+3):(j+3+h+1)].to_numpy();

            tvec_i = np.arange(k, k+h+1)

            # Find relevant input at the specific hours of flexibility
            z_i = z[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1)
            if perfectForesight:
                u_forecast = u[tvec_i]
            u_t_true = u[k]

            l = dfp['l_hours_avail'][i] #l = 36 # slet
            
            # Solve
            prob, x_d, b, x_s = StochasticProgram(scenarios, n_scenarios, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, verbose=False)
            if LpStatus[prob.status] != 'Optimal':
                print("\n\nPlugged in = ", z[k], z_i[0])
                print("bmin = ", bmin[k], bmin_i[0])
                print("u = ", u[k], u_forecast[0])
                print("b0 = ", value(b[1,0]))
                print("x = ", value(x_d[0]), "Trying ", value(x_d[0])+value(b[1,0]), " <= ", bmax)
                print("Infeasible at k = " + str(k) + " with i = " + str(i) + " and j = " + str(j))
                print("\n\n\n")

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
h = 4*24 # 5 days horizon for the multi-day smart charge
prob, x, b = MultiDayStochastic(scenarios, n_scenarios, dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24)
plot_EMPC(prob, 'Stochastic Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
