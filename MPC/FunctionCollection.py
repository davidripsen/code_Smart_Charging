"""
Collection of functions for Smart Chargng
"""
from pulp import *
import numpy as np
import plotly.graph_objects as go

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
        prob.solve()
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Return results
    return(prob, x, b)

def ImperfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u_t_true, u_forecast, z, T, tvec, r, verbose=True):
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
        prob += b[t+1] == b[t] + x[t]*r - u_forecast[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t+1] <= bmax
        prob += x[t] <=   xmax * z[t]
        prob += x[t] >= 0

    # Solve problem
    if verbose:
        prob.solve()
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Update b1 with actual use (relative to what we chose to charge)
    b[1] = b[0] + x[0] - u_t_true
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
    fig.update_layout(title=name + "    MPC of EVs        from " + starttime +" to "+ endtime+"      Total cost: " + str(round(obj)) + " DKK  (+tariffs)",
        xaxis_title="Days",
        yaxis_title="kWh or DKK/kWh")
    fig.show()

    ## Export figure
    if export:
        fig.write_html( "plots/MPC/" + name + "_mpc.html")

def DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, r=1):
    # Init problem
    prob = LpProblem("mpc1", LpMinimize)

    # Init variables
    global x
    global b
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
    i = LpVariable.dicts("i", tvec, lowBound=0, upBound=1, cat='Binary')
    b[0] = b0
    M = 10**6

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]))

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t]*r - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t+1] <= bmax
        
        ######## DUMB CHARGE ########
        ### Implement in OR-terms: x[t] == min(z[t]*xmax, bmax-b[t])
        #######
        # Ensure i[t] == 1, if z[t]*xmax < bmax-b[t] (dvs. i=1 når der er rigeligt plads på batteriet)
        prob += bmax-b[t] - z[t]*xmax  - M*i[t] <= 0
        prob += z[t]*xmax - bmax+b[t] - M*(1-i[t]) <= 0

        # Use i[t] to constraint x[t]
        prob += x[t] <= z[t]*xmax
        prob += x[t] <= bmax-b[t]
        prob += x[t] >= (z[t]*xmax - M*(1-i[t])) # i = 1 betyder, at der lades max kapacitet
        prob += x[t] >= (bmax-b[t] - M*i[t])     # i = 0 betyder, at vi kun kan lade de resterende til 100 % SOC
        #prob += i[t] <= z[t]

    # Solve problem
    prob.solve()

    # Return results
    return(prob, x, b)