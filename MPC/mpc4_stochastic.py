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
from sklearn_extra.cluster import KMedoids
from code_Smart_Charging.MPC.FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge, ExtractEVdataForMPC, MultiDay
pd.set_option('display.max_rows', 500)
runMany = True
runDeterministicReference = True

# Read scenarios from txt
scenarios = np.loadtxt('./data/MPC-ready/scenarios.csv', delimiter=','); scenarios_all=scenarios;

# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)


i = 0 # i=2 Good performance (from stochastic model), i=3: Shitty performance
dfv, dfspot, dfp, dft, timestamps, z, u, uhat, b0, r, bmin, bmax, xmax, c_tilde, vehicle_id, firsthour, starttime, endtime = ExtractEVdataForMPC(dfv=DFV[i], z_var='z_plan_everynight', u_var='use_lin',
                                                                                                                                                uhat_var='use_org_rolling', bmin_var='SOCmin_everymorning', p=0.10)


#################################################### LET'S GO! ########################################################


# Note: Scenarios as is right now, do not take into account that the uncertainty/scenarios differences are very dependent of time of day.
def StochasticProgram(scenarios, n_scenarios, h, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l, previous_solution=None, KMweights=None, verbose=True):
    """
    Solves the 2-stage stochastic program for a given time horizon T, and returns the optimal solution.
    l: Length of deterministic prices
    O: Number of scenarios (Omega)
    """
    scenarios = scenarios[0:n_scenarios, :] # for Dev: Antag n_scenarier scenarier
    #scenarios = scenarios_all
    if KMweights is None:
        KMweights = np.repeat(1/n_scenarios, n_scenarios)
    O, K = scenarios.shape
    tvec_d = tvec[0:l] # Deterministic part
    tvec_s = tvec[l:]  # Stochastic part
    c_d = c_forecast[:l] # Deterministic part
    c_s = c_forecast + scenarios[:, :(h+1)] # Stochastic part
    c_s[c_s < 0] = 0 # Truncate cost_stochastic to assume non-negative electricity spot prices

    ### Init problem
    prob = LpProblem("StochEcoMPC", LpMinimize)

    ### Init variables
    x_d = LpVariable.dicts("x_d", tvec_d, lowBound=0, upBound=xmax, cat='Continuous')
    x_s = LpVariable.dicts("x_s", [(t,o) for o in range(O) for t in tvec_s], lowBound=0, upBound=xmax, cat='Continuous') #xs_i,omega
    b = LpVariable.dicts("b", [(t,o) for o in range(O) for t in np.append(tvec,tvec[-1]+1)], lowBound=0, upBound=5000, cat='Continuous')
    s = LpVariable.dicts("s", [(t,o) for o in range(O) for t in tvec], lowBound=0, upBound=0.25*bmax, cat='Continuous') # Add penalizing slack for violating bmax=80%, but still remain below 100%
    s2 = {(i, o): LpVariable("s2_("+str(i)+",_"+str(o)+")", lowBound=0, upBound=ub) for o in range(O) for i, ub in enumerate(bmin)}
    # Set initial SOC to b0 for all scenarios o
    for o in range(O): b[(0,o)] = b0

    # # warm-start
    # if (previous_solution is not None) and (l > 1):
    #     x_d_prev = previous_solution[0]
    #     x_s_prev = previous_solution[1]
    #     for t in range(1, len(x_d_prev)):
    #         if t < l:
    #             x_d[t-1].setInitialValue(round(x_d_prev[t].value(),5))
    #     for t in range(l+1, l+int(len(x_s_prev)/O)):
    #         if t <= h:
    #             for o in range(O):
    #                 x_s[t-1,o].setInitialValue(round(x_s_prev[(t,o)].value(), 5))

    ### Objective
    prob += lpSum([c_d[t]*x_d[t] for t in tvec_d]) + lpSum([KMweights[o] * c_s[o,t]*x_s[t,o] for t in tvec_s for o in range(O)]) - lpSum([KMweights[o] * c_tilde * ((b[tvec[-1],o]) - b[0,o]) for o in range(O)]) + lpSum([KMweights[o] * 100*O*c_tilde*(s[t,o]+s2[t+1,o]) for t in tvec for o in range(O)])

    ### Constraints
        # Deterministic part
    for t in tvec_d:
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_d[t]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1] - s2[t+1,o]
            prob += b[(t+1,o)] <= bmax + s[t,o]
            prob += x_d[t] <= xmax * z[t]
            prob += x_d[t] >= 0

        # Stochastic part$
    for t in tvec_s:
        for o in range(O):
            prob += b[(t+1,o)] == b[(t,o)] + x_s[(t,o)]*r - u_forecast[t]
            prob += b[(t+1,o)] >= bmin[t+1] - s2[t+1,o]
            prob += b[(t+1,o)] <= bmax + s[t,o]
            prob += x_s[(t,o)] <= xmax * z[t]
            prob += x_s[(t,o)] >= 0

    # Solve problem
    if verbose:
        prob.solve(PULP_CBC_CMD(warmStart=(previous_solution != None)))
    else:
        prob.solve(PULP_CBC_CMD(msg=0, warmStart= (previous_solution != None)))
        print("Status:", LpStatus[prob.status])

    #Update b1 with actual use (relative to what we chose to charge) (Should be sufficient only to update b(1,0))
    for o in range(O):
        b[(1,o)] = b[(0,o)] + value(x_d[0]) - u_t_true
        prob.assignVarsVals({'b_(1,_'+str(o)+')': b[1,o]})
        assert b[1,o] == value(b[1,0]), "b(1,o) is not equal to value(b(1,0))"
        # ^ Most of this code is redundant

    # Return results
    return(prob, x_d, b, x_s)

# prob, x_d, b, x_s = StochasticProgram(scenarios, 10, b0, bmax, bmin, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z, tvec, r, l=12, verbose=True)

# # Print results nicely
# print("Status:", LpStatus[prob.status])
# print("Objective:", value(prob.objective))
# for v in prob.variables():
#     print(v.name, "=", v.varValue)


# # Plot scenarios
# plt.figure()
# for j in range(0, n_scenarios):
#     #plt.plot(c_forecast + scenarios[j,:], label='Scenario '+str(j))
#     plt.plot(c_s[j,:], label='Scenario '+str(j))
# plt.plot(c_forecast, label='Forecasted Price', color='black', linewidth=3)
# c = dft.iloc[i, (j+3):(j+3+maxh+1)].to_numpy();
# plt.plot(c, label='True Price', color='black', linewidth=3)
# plt.ylim(-0.5, 7)
# plt.plot(c, label='True Price')
# plt.show()

# Function to extract the mediods of performing KMediods clustering on the scenarios using sklearn_extra.cluster.KMedoids
def getMediods(scenarios, n_clusters):
    # Perform KMedoids clustering
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(scenarios)
    # Extract mediods
    mediods = scenarios[kmedoids.medoid_indices_]
    # Extract proportion of scenarios in each cluster
    cluster_proportions = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_proportions[i] = np.mean(kmedoids.labels_ == i)
    # Return mediods and cluster proportions
    return(mediods, cluster_proportions)


def MultiDayStochastic(scenarios, n_scenarios, dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, KMweights=None, maxh=6*24, perfectForesight=False, verbose=False):
    # maxh = 6*24 # Delete
    # h = 4*24 # Delete
    # scenarios = mediods # Delete
    # KMweights = KMweights # Delete
    # n_scenarios = len(KMweights) # Delete
    # perfectForesight=False

    # Study from first hour of prediciton up to and including the latest hour of known spot price
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
    # perfectForesight = False # Deleter

    #L = 200 # Delete


    # Init
    prev_sol = None
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan
    c = dfspot['TruePrice'].to_numpy()
    costs = 0
    k = 0
    # E.g.  k, i, j, l, b0 = 163, 163, 0, 17, 60.17055119267742
    
    # For each Atime
    for i in range(len(dfp)):
        # For each hour until next forecast
        for j in range(dfp['Atime_diff'][i]):
            if k%50 == 1: print("k = " + str(k) + " of " + str(L-1))
            # Extract forecasts from t=0..h
            c_forecast = dfp.iloc[i, (j+3):(j+3+h+1)].to_numpy();
            if perfectForesight:
                c_forecast = dft.iloc[i, (j+3):(j+3+h+1)].to_numpy();

            # Find relevant input at the specific hours of flexibility
            tvec_i = np.arange(k, k+h+1)
            z_i = z[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1)
            if perfectForesight:
                u_forecast = u[tvec_i]
            u_t_true = u[k]

            l = max(1, dfp['l_hours_avail'][i]-j) # If (we due to missing forecasts/data) run out of known-hours, only treat from second hour and onwards as stochastic. There is 0 variance on that though, so....
            
            # Solve
            prob, x_d, b, x_s = StochasticProgram(scenarios, n_scenarios, h, b0, bmax, bmin_i, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z_i, tvec, r, l, previous_solution=None, KMweights=KMweights, verbose=verbose)
            if LpStatus[prob.status] != 'Optimal':
                print("\n\nPlugged in = ", z[k],"=", z_i[0])
                print("bmin = ", round(bmin[k]), round(bmin_i[0]), "bmin_t+1 = ", round(bmin_i[1]))
                print("u_true, u_forecast = ", u[k], u_forecast[0])
                print("b0 = ", b0, "b1 = ", value(b[1,0]))
                print("x = ", value(x_d[0]), "Trying  ", bmin[k+1],"<=", r*value(x_d[0])+b[0,0]-u[k], " <= ", bmax)
                print("Infeasible at k = " + str(k) + " with i = " + str(i) + " and j = " + str(j), " and l = " + str(l))
                print("\n\n\n")

            # Implement/store only the first step, and re-run in next hour
            x0 = value(x_d[0]); X[k]=x0;                # Amount charged in the now-hour
            b1 = value(b[1,0]); B[k+1]=b1;              # Battery level after the now-hsecour / beggining of next hour
            costs += x0 * c[k];                         # Cost of charging in the now-hour
            b0 = b1                                     # Next SOC start is the current SOC
            k += 1
            prev_sol = [x_d, x_s]                       # For warm-start

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'z':z[0:L], 'objective':total_cost}
                return(prob, X, B)

### Run the problem
h = 4*24 # 5 days horizon for the multi-day smart charge
n_scenarios = 10
#prob, x, b = MultiDayStochastic(scenarios_all, n_scenarios, dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, KMweights=None, maxh = 6*24)
#plot_EMPC(prob, 'Stochastic Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)

#### Run the problem on mediods
n_clusters=10
mediods, weights = getMediods(scenarios_all, n_clusters=n_clusters)
h = 4*24 # 5 days horizon for the multi-day smart charge
prob_stochKM, x, b = MultiDayStochastic(mediods, n_clusters, dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, KMweights=weights, maxh = 6*24)
plot_EMPC(prob_stochKM, 'Stochastic Multi-Day (+kMediods) Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
#### Evt missing: Implement warmstart

if runDeterministicReference:
    # ### Multi-Dayahead (Deterministic)
    # h = 4*24 # 5 days horizon for the multi-day smart charge
    # prob_mda, x, b = MultiDay(dfp, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24, perfectForesight=False)
    # #prob, x, b = MultiDay(dft, dfspot, u, uhat, z, 6*24, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24) # Snyd: kendte priser
    # plot_EMPC(prob_mda, 'Multi-Day Smart Charge (h = '+str(int(h/24))+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)

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

    ### Day-Ahead SmartCharge
    prob_da, x, b = MultiDay(dfp, dfspot, u, uhat, z, 0, b0, bmax, bmin, xmax, c_tilde, r, DayAhead=True, maxh=6*24, perfectForesight=False)
    plot_EMPC(prob_da, 'Day-Ahead Smart Charge of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax, firsthour=firsthour)


    ### Perfect Foresight
    prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r, verbose=True)
    plot_EMPC(prob, 'Perfect Foresight   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within, z_within,  starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
        # Verify the objective value
    print("Objective value = ", prob.objective.value())
    print("Objective value = ", np.sum([value(x[t]) * c_within[t] for t in tvec_within]) - c_tilde * (value(b[T+1]) - b[0]))

    ### DumbCharge
    prob_dc, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r=r, verbose=False)
    plot_EMPC(prob_dc, 'Dumb Charge   of vehicle = ' + str(vehicle_id) + '   r = '+str(r), x, b, u_within, c_within, z_within, starttime=str(starttime.date()), endtime=str(endtime.date()), export=False, BatteryCap=bmax, firsthour=firsthour)
    

# Conclusion. One would expect that the total costs for the models are  Perfect Foresight < Stochastic + kMediods  < Stochastic <   MultiDay  <  Day-Ahead Smart Charge      (EXPECTED)
# However, it is (with h=4 days) # for DFV[3]                           Perfect Foresight < Day-Ahead < Stochastic < Stochastic + kMediods    < MutliDay                     (ACTUAL  )
# <=> VERY DIFFERENT :((


# # Visualise mediods
# fig, ax = plt.subplots(1,1, figsize=(10,5))
# ax.plot(mediods.T)
# ax.set_title(str(n_clusters) + ' Mediods')
# ax.set_xlabel('Hour')
# ax.set_ylabel('Price')
# plt.show()

# ### Cumsum of costs for different models
# costs_stochKM = np.cumsum(prob_stochKM['x'] * prob_stochKM['c'])
# costs_da = np.cumsum(prob_da['x'] * prob_da['c'])

#     # Visualise
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(costs_stochKM)), y=costs_stochKM, name='Stochastic + kMediods'))
# fig.add_trace(go.Scatter(x=np.arange(len(costs_da)), y=costs_da, name='Day-Ahead'))
# fig.update_layout(title='Cumulative costs for different models    of vehicle = '+str(vehicle_id), xaxis_title='Time', yaxis_title='Costs')
# fig.show()