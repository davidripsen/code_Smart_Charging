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
from code_Smart_Charging.MPC.FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge, PlotChargingProfile, ExtractEVdataForMPC
#from code_Smart_Charging.dataviz_cardata2 import PlotChargingProfile
pd.set_option('display.max_rows', 500)
runMany = True

# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Use
dfv, dfspot, dfp, dft, timestamps, z, u, uhat, b0, r, bmin, bmax, xmax, c_tilde, vehicle_id, firsthour, starttime, endtime = ExtractEVdataForMPC(dfv=DFV[3], z_var='z_plan_everynight', u_var='use_lin',
                                                                                                                                                 uhat_var='use_org_rolling', bmin_var='SOCmin_everymorning', p=0.10)


#################################################### LET'S GO! ########################################################


#### Tasks:
# Modify function such that bmax can be a series, not just a scalar
def MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, DayAhead=False, maxh=6*24, perfectForesight=False):
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
            if k%50 == 0:
                print("k = " + str(k) + " of " + str(L-1))
            
            if DayAhead:  # If Day-Ahead Smart Charge, disregard h input and use h = l_hours_avail
                h = dfp['l_hours_avail'][i]-1-j
                if h<0: 
                    h=0# Account for missing forecasts
                    print("Missing forecasts at k=",k,"i=",i,"j=",j, "... Setting h=0")
                tvec = np.arange(0,h+1)

            # Extract forecasts from t=0..h
            c_forecast = dfp.iloc[i, (j+3):(j+3+h+1)].to_numpy()
            if perfectForesight:
                c_forecast = dft.iloc[i, (j+3):(j+3+h+1)].to_numpy()
            
            # Find relevant input at the specific hours of flexibility
            tvec_i = np.arange(k, k+h+1)
            z_i = z[tvec_i] # Assuming known plug-in times.
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1) # = actually uhat[k-1], but a 0 has been appended as first value.
            if perfectForesight:
                u_forecast = u[tvec_i]
            u_t_true = u[k]
            

            # Solve
            prob, x, b = ImperfectForesight(b0, bmax, bmin_i, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z_i, h, tvec, r, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
            #print("Status:", LpStatus[prob.status])
            if LpStatus[prob.status] != 'Optimal':
                print("\n\nPlugged in = ", z[k],"=", z_i[0])
                print("bmin = ", round(bmin[k]), round(bmin_i[0]), "bmin_t+1 = ", round(bmin_i[1]))
                print("u = ", u[k], u_forecast[0])
                print("b0 = ", b0, "b1 = ", value(b[1]))
                print("x = ", value(x[0]), "Trying  ", bmin[k+1],"<=", r*value(x[0])+b0-u[k], " <= ", bmax)
                print("Infeasible at k = " + str(k) + " with i = " + str(i) + " and j = " + str(j))
                print("\n\n\n")
            
            # Implement/store only the first step, and re-run in next hour
            x0 = value(x[0]); X[k]=x0;                # Amount charged in the now-hour
            b1 = value(b[1]); B[k+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * c[k];                       # Cost of charging in the now-hour
            b0 = b1                                   # Next SOC start is the current SOC
            k += 1

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'z':z[0:L], 'objective':total_cost}
                return(prob, X, B)

### Run the problem
if not runMany:
    h = 4*24 # 5 days horizon for the multi-day smart charge
    prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24, perfectForesight=False)
    #prob, x, b = MultiDay(dft, dfspot, u, uhat, z, 6*24, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24) # Snyd: kendte priser
    plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)


#################################################### RUN ALL THE MODELS ########################################################

# Compare models on the data within horizon
maxh = 6*24
L = len(u) - (maxh+1)
T = L - 1
tvec = np.arange(T+1)
T_within = T #-maxh 
c_within = dfspot['TruePrice'][0:T_within+1] # Actually uses all prices in this case:-)
tvec_within = tvec[0:T_within+1]
z_within = z[0:T_within+1]
u_within = u[0:T_within+1]
u2_within = dfv['use'].to_numpy()[0:T_within+1]
bmin_within = bmin[0:T_within+2]

### Perfect Foresight
prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r, verbose=True)
plot_EMPC(prob, 'Perfect Foresight   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within, z_within,  starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
    # Verify the objective value
print("Objective value = ", prob.objective.value())
print("Objective value = ", np.sum([value(x[t]) * c_within[t] for t in tvec_within]) - c_tilde * (value(b[T+1]) - b[0]))

### Day-Ahead SmartCharge
prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, 0, b0, bmax, bmin, xmax, c_tilde, r, DayAhead=True, maxh=6*24, perfectForesight=True)
plot_EMPC(prob, 'Day-Ahead Smart Charge of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax, firsthour=firsthour)

### Multi-Day SmartCharge
if runMany:
    # for h in [i for i in [0]]: # Hour-ahead Smart Charging
    #     print("h = " + str(h))
    #     prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh=6*24, perfectForesight=False) ### FUUUUCK
    #     plot_EMPC(prob, 'Hours-ahead Smart Charge (h = '+str(h)+' hours) of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax, firsthour=firsthour)
    #     print("Total cost: " + str(prob['objective']))
    #     print("")

    for h in [i*24 for i in range(0,7)]: # Day-ahead Smart Charging
        print("h = " + str(h))
        prob, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh=6*24, perfectForesight=False)
        plot_EMPC(prob, 'Multi-Day Smart Charge (h = '+str(int(h/24))+' days) of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax, firsthour=firsthour)
        print("Total cost: " + str(prob['objective']))
        print("")

### DumbCharge
prob, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r=r, verbose=False)
if LpStatus[prob.status] == 'Optimal':
    plot_EMPC(prob, 'Dumb Charge   of vehicle = ' + str(vehicle_id) + '   r = '+str(r), x, b, u_within, c_within, z_within, starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
        # Verify the objective value
    print("Objective value = ", prob.objective.value())
    print("Objective value = ", np.sum([value(x[t]) * c_within[t] for t in tvec_within]) - c_tilde * (value(b[T+1]) - b[0]))
else:
    print("\n\n DumbCharge failed on this set of data (vehicle=" + str(vehicle_id) + ") \n\n")


# ### Actual charge
PlotChargingProfile(dfvehicle=DFV[i], plot_efficiency_and_SOCmin=False) # Remember to define function


# x = dfv['charge'][:-maxh]
# c = dfv['trueprice'][:-maxh]
# b = dfv['SOC'][:-maxh]
# totalcost = (x * c).sum()  -  c_tilde * b.iloc[-1] - b[0]
# print("Total cost of actual charging: " + str(totalcost))
# plt.plot(dfv['use_lin'][:-maxh])
# plt.title('Total cost: ' + str(totalcost))
# plt.show()
# # Beware, that the counterfactual and the factual are not directly comparable.
# #   - Particularly due to median(efficiency) vs. actual efficiency is used
# #   - Counterfactual plug-ins
# #   - bmin vs. only a bmax              (not a direct problem for comparison)
# # MontaSmartCharge typically boils down to a mix between DumbCharge and DayAheadSmartCharge.
