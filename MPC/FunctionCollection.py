"""
Collection of functions for Smart Chargng
"""
from pulp import *
import numpy as np
import plotly.graph_objects as go
import datetime
import pandas as pd
from sklearn_extra.cluster import KMedoids

def PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, r=1, verbose=True):
    # Init problem 
    prob = LpProblem("mpc1", LpMinimize)

    # Init variables
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax*1.25, cat='Continuous')
    s = LpVariable.dicts("s", tvec, lowBound=0, upBound=0.25*bmax, cat='Continuous')
    s2 = {i: LpVariable("s2_"+str(i), lowBound=0, upBound=ub, cat='Continuous') for i, ub in enumerate(bmin)}
    b[0] = b0

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]) + [100*c_tilde*(s[t]+s2[t+1]) for t in tvec])

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t]*r - u[t]
        prob += b[t+1] >= bmin[t+1] - s2[t+1]
        prob += b[t+1] <= bmax + s[t]
        prob += x[t] <= xmax * z[t]
        prob += x[t] >= 0

    # Solve problem
    if verbose:
        prob.solve(PULP_CBC_CMD(msg=1))
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Return objective without penalization
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]))

    # Return results
    return(prob, x, b)

def ImperfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u_t_true, u_forecast, z, T, tvec, r, verbose=False):
    # Init problem 
    prob = LpProblem("mpc1", LpMinimize)

    # Init variabless
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=5000, cat='Continuous')
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

def plot_EMPC(prob, name="", x=np.nan, b=np.nan, u=np.nan, c=np.nan, z=np.nan, starttime='', endtime='', export=False, export_only = False, BatteryCap=60, firsthour=0, vehicle_id=''):
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
    if not export_only:
        fig.show()

    ## Export figure
    if export:
        if vehicle_id != '':
            # Make vehicle_unique folder
            vehicle_id = str(vehicle_id) + "/"
            if not os.path.exists("plots/MPC/"+vehicle_id):
                os.makedirs("plots/MPC/"+vehicle_id)
        fig.write_html( "plots/MPC/"+vehicle_id + name + "_mpc.html")

def DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, r=1, verbose=False):
    # Init problem
    prob = LpProblem("mpc_DumbCharge", LpMinimize)

    # Init variables
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=5000, cat='Continuous')
    i = LpVariable.dicts("i", tvec, lowBound=0, upBound=1, cat='Binary')
    s = LpVariable.dicts("s", tvec, lowBound=0, upBound=0.25*bmax, cat='Continuous')
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
        prob += x[t] >= (bmax+s[t]-b[t]) / r - M*i[t]   # i = 0 betyder, at vi kun kan lade de resterende til 80 eller 100 % SOC
        #prob += i[t] <= z[t]

    # Solve problem
    prob.solve(PULP_CBC_CMD(gapAbs = 0.01, msg=verbose))

    # Return objective without penalization
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]))

    # Return results
    return(prob, x, b)

### Stochastic programming functions. Maintained in here
def StochasticProgram(n_scenarios, b0, bmax, bmin, xmax, c_d, c_s, c_tilde, u_t_true, u_forecast, z, tvec, r, l, previous_solution=None, KMweights=None, verbose=True):
    """
    Solves the 2-stage stochastic program for a given time horizon T, and returns the optimal solution.
    l: Length of deterministic prices
    O: Number of scenarios (Omega)
    """

    if KMweights is None:
        KMweights = np.repeat(1/n_scenarios, n_scenarios)
    O, K = c_s.shape
    tvec_d = tvec[0:l] # Deterministic part
    tvec_s = tvec[l:]  # Stochastic part
    
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
        b[(1,o)] = b[(0,o)] + value(x_d[0])*r - u_t_true
        prob.assignVarsVals({'b_(1,_'+str(o)+')': b[1,o]})
        assert b[1,o] == value(b[1,0]), "b(1,o) is not equal to value(b(1,0))"
        # ^ Most of this loop code is redundant

    # Return results
    return(prob, x_d, b, x_s)

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

# Maintained here
def MultiDayStochastic(scenarios, n_scenarios, dfp, dft, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, KMweights=None, maxh=6*24, perfectForesight=False, verbose=False):
    
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
    H = h; # Store h
    # perfectForesight = False

    # Init
    flag_AllFeasible = True
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
        h = H
        l = dfp['l_hours_avail'][i]+1
        # For each hour until next forecast
        for j in range(dfp['Atime_diff'][i]):
            if k%50 == 0:
                print("k = " + str(k) + " of " + str(L-1))
            
            # Patch holes in forecasts (1 out of 2)
            l = l-1
            if l < 12: # New prices are known at 13.00
                l = 35

            # When re-using the same forecast, shorten the horizon
            if j>0:
                h = max(h-1, l-1) # h = h-1 but don't go below the DayAhead horizon
            h = min(h, L-k) # Allow control to know that experiment is ending.
            tvec = np.arange(0,h+1)
            #print("i,j,k,l,h = ", i,j,k,l,h)

            # Extract forecasts from t=0..h
            c_forecast = dfp.iloc[i, (j+3):(3+H+1)].to_numpy();
            if perfectForesight:
                c_forecast = dft.iloc[i, (j+3):(3+H+1)].to_numpy();
            
            # Patch holes in forecasts (2 out of 2) - use known prices
            c_forecast[:min(l,h+1)] = dft.iloc[i, (j+3):(3+H+1)].to_numpy()[:min(l,h+1)]

            # Extract deterministic and stochastic prices
            idx = np.random.randint(0, scenarios.shape[0]-n_scenarios)
            scenarioExtract = scenarios[idx:idx+n_scenarios, :] # Subset new scenarios every iteration
            c_d = c_forecast[:l] # Deterministic part
            c_s = c_forecast + scenarioExtract[:, j:(H+1)] # Stochastic part
            c_s[c_s < 0] = 0 # Truncate cost_stochastic to assume non-negative electricity spot prices. Conclussion: Performed better.

            # Find relevant input at the specific hours of flexibility
            tvec_i = np.arange(k, k+h+1)
            z_i = z[tvec_i]
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1)
            if perfectForesight:
                u_forecast = u[tvec_i]
            u_t_true = u[k]

            # Solve
            if z_i[0] != 0: # Plugged in
                prob, x_d, b, x_s = StochasticProgram(n_scenarios, b0, bmax, bmin_i, xmax, c_d, c_s, c_tilde, u_t_true, u_forecast, z_i, tvec, r, l, previous_solution=None, KMweights=KMweights, verbose=verbose)
                if LpStatus[prob.status] != 'Optimal':
                    flag_AllFeasible = False
                    print("\n\nPlugged in = ", z[k],"=", z_i[0])
                    print("bmin = ", round(bmin[k]), round(bmin_i[0]), "bmin_t+1 = ", round(bmin_i[1]))
                    print("u_true, u_forecast = ", u[k], u_forecast[0])
                    print("b0 = ", b0, "b1 = ", value(b[1,0]))
                    print("x = ", value(x_d[0]), "Trying  ", bmin[k+1],"<=", r*value(x_d[0])+b[0,0]-u[k], " <= ", bmax)
                    print("Infeasible at k = " + str(k) + " with i = " + str(i) + " and j = " + str(j), " and l = " + str(l))
                    print("\n\n\n")
                x0 = value(x_d[0])
                b1 = value(b[1,0])
            elif z_i[0] == 0: # Not plugged in
                x0 = 0
                b1 = b0 + x0*r - u_t_true

            # Implement/store only the first step, and re-run in next hour
            X[k]=x0;                # Amount charged in the now-hour
            B[k+1]=b1;              # Battery level after the now-hsecour / beggining of next hour
            costs += x0 * c[k];                         # Cost of charging in the now-hour
            b0 = b1                                     # Next SOC start is the current SOC
            k += 1
            #prev_sol = [x_d, x_s]                       # For warm-start

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'z':z[0:L], 'objective':total_cost}
                return(prob, X, B, flag_AllFeasible)

# Maintained here (from mpc3_montadata.py)
def MultiDay(dfp, dft, dfspot, u, uhat, z, h, b0, bmax, bmin, xmax, c_tilde, r, DayAhead=False, maxh=6*24, perfectForesight=False):
    # Study from first hour of prediciton up to and including the latest hour of known spot price
    L = len(u) - (maxh+1) # Run through all data, but we don't have forecasts of use/plug-in yet.
                        # maxh = maximum h of interest ==> to allow comparison on exact same data for different horizons h.
    H = h # store h
    # Init
    flag_AllFeasible = True
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan
    c = dfspot['TruePrice'].to_numpy()
    costs = 0
    k = 0

    # For each Atime
    for i in range(len(dfp)):
        h = H
        tvec = np.arange(0,h+1)
        flagForecastHole = 0
        l = dfp['l_hours_avail'][i]+1
        # For each hour until next forecast
        for j in range(dfp['Atime_diff'][i]):
            if k%50 == 0:
                print("k = " + str(k) + " of " + str(L-1))
            
            # Patch holes in forecasts (1 out of 2)
            l = l-1
            if l < 12: # New prices are known at 13.00
                l = 35
                flagForecastHole += 1

            if DayAhead:  # If Day-Ahead Smart Charge, disregard h input and use h = l_hours_avail-1
                h = l-1
                #H = dfp['l_hours_avail'][i]-1
                H = dfp['l_hours_avail'][i]-1 + 24*flagForecastHole
                #print("i,j,k,l,h = ", i,j,k,l,h)

            # When re-using the same forecast, shorten the horizon
            if (j>0) and (not DayAhead):
                h = max(h-1, l-1) # h = h-1 but don't go below the DayAhead horizon
            h = min(h, L-k) # Allow control to know that experiment is ending.
            tvec = np.arange(0,h+1)

            # Extract forecasts from t=0..h
            #c_forecast = dfp.iloc[i, (j+3):(j+3+h+1)].to_numpy()
            c_forecast = dfp.iloc[i, (j+3):(3+H+1)].to_numpy()
            if perfectForesight:
                c_forecast = dft.iloc[i, (j+3):(3+H+1)].to_numpy();
                
            # Patch holes in forecasts (2 out of 2) - use known prices
            c_forecast[:min(l,h+1)] = dft.iloc[i, (j+3):(3+H+1)].to_numpy()[:min(l,h+1)]
            
            # Find relevant input at the specific hours of flexibility
            tvec_i = np.arange(k, k+h+1)
            z_i = z[tvec_i] # Assuming known plug-in times.
            bmin_i = bmin[np.append(tvec_i, tvec_i[-1]+1)]

            u_forecast = np.repeat(uhat[k], h+1) # = actually uhat[k-1], but a 0 has been appended as first value.
            if perfectForesight:
                u_forecast = u[tvec_i]
            u_t_true = u[k]
            

            # Solve
            if z_i[0] != 0:
                prob, x, b = ImperfectForesight(b0, bmax, bmin_i, xmax, c_forecast, c_tilde, u_t_true, u_forecast, z_i, h, tvec, r, verbose=False) # Yes, it is tvec=0..h, NOT tvec_i
                #print("Status:", LpStatus[prob.status])
                if LpStatus[prob.status] != 'Optimal':
                    flag_AllFeasible = False
                    print("\n\nPlugged in = ", z[k],"=", z_i[0])
                    print("bmin = ", round(bmin[k]), round(bmin_i[0]), "bmin_t+1 = ", round(bmin_i[1]))
                    print("u = ", u[k], u_forecast[0])
                    print("b0 = ", b0, "b1 = ", value(b[1]))
                    print("x = ", value(x[0]), "Trying  ", bmin[k+1],"<=", r*value(x[0])+b0-u[k], " <= ", bmax)
                    print("Infeasible at k = " + str(k) + " with i = " + str(i) + " and j = " + str(j))
                    print("\n\n\n")
                x0 = value(x[0])
                b1 = value(b[1])
            elif z_i[0] == 0: # Not plugged in
                x0 = 0
                b1 = b0 + x0*r - u_t_true

            # Implement/store only the first step, and re-run in next hour
            X[k]=x0;                # Amount charged in the now-hour
            B[k+1]=b1;              # Battery level after the now-hour / beggining of next hour
            costs += x0 * c[k];                       # Cost of charging in the now-hour
            b0 = b1                                   # Next SOC start is the current SOC
            k += 1

            # THE END
            if k == L:
                # Costs
                total_cost = np.sum(costs) - c_tilde * (B[-1] - B[0])

                # Tie results intro prob
                prob = {'x':X, 'b':B, 'u':u[0:L], 'c':c[0:L], 'z':z[0:L], 'objective':total_cost}
                return(prob, X, B, flag_AllFeasible)

# Maitained here
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
    #bmax = dfv['SOCmax'].median()
    bmax = 0.8*dfv['BatteryCapacity'].median()
    #bmax = np.nanmin([dfv['SOCmax'], dfv['BatteryCapacity']], axis=0)
    xmax = dfv['CableCapacity'].unique()[0]
    c_tilde = np.quantile(dfspot['TruePrice'], p) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

    return dfv, dfspot, dfp, dft, timestamps, z, u, uhat, b0, r, bmin, bmax, xmax, c_tilde, vehicle_id, firsthour, starttime, endtime

# Maintained in dataviz_cardata2.py
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