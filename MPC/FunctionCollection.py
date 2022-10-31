"""
Collection of functions for Smart Chargng
"""
from pulp import *
import numpy as np
import plotly.graph_objects as go

def PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, verbose=True):
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
        prob += b[t+1] == b[t] + x[t] - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t+1] <= bmax
        prob += x[t] <= xmax*z[t]
        prob += x[t] >= 0
                # Debugging tips: Du kan ikke constrainte en variabels startpunkt, når startpunktet har fået en startværdi.

    # Solve problem
    if verbose:
        prob.solve()
    else:
        prob.solve(PULP_CBC_CMD(msg=0))

    # Return results
    return(prob, x, b)

def plot_EMPC(prob, name="", x=np.nan, b=np.nan, u=np.nan, c=np.nan, starttime='', endtime='', export=False):
        # Identify iterative-appended, self-made prob
    fig = go.Figure()
    if type(prob) == dict:
        tvec = np.arange(0, len(prob['x']))
        tvec_b = np.arange(0, len(prob['b']))
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(prob['b'][t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['x'][t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['u'][t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['c'][t]) for t in tvec], mode='lines', name='True Price'))
        obj = prob['objective']
    else:
        tvec = np.arange(0, len(x))
        tvec_b = np.arange(0, len(b))
        obj = value(prob.objective)
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(b[t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='True Price'))
    
    fig.update_xaxes(tickvals=tvec_b[::24], ticktext=[str(t//24) for t in tvec_b[::24]])
    # Fix y-axis to lie between 0 and 65
    fig.update_yaxes(range=[-3, 62])

    # add "Days" to x-axis
    # Add total cost to title
    fig.update_layout(title=name + "    MPC of EVs (simulated consumer data)        from " + starttime +" to "+ endtime+"      Total cost: " + str(round(obj)) + " DKK (+tariffs)",
        xaxis_title="Days",
        yaxis_title="kWh or DKK/kWh")
    fig.show()

    ## Export figure
    if export:
        fig.write_html( "plots/MPC/" + name + "_mpc.html")