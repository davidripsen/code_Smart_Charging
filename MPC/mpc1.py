"""
    Implementation of the economic MPC problem for multi-day Smart Charging of EVs.
"""
# Imports
from pulp import *
import numpy as np

##################### TEMPORARY SIMULATION OF VAR #################
# Horizon
T = 24*14
tvec = np.arange(0,T+1)

# External variables (SIMULATED)
plugin = 17.25; plugout = 7.15;
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at t
c = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
c_tilde = min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h
u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1)

# Parameters of the battery
battery_size = 60 # kWh
b0 = 0.8 * battery_size
bmax = 1 * battery_size
xmax = 7  # kW (max charging power)

# User input
bmin_morning = 0.40 * battery_size;
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])


##################################################################
################## IMPLEMENTATION OF THE PROBLEM ##################
# Init problem
prob = LpProblem("mpc1", LpMinimize)

# Init variables
x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
b[0] = b0

# Objective
prob += lpSum([c[t]*x[t] for t in tvec] + c_tilde * b[T+1])

# Constraints
for t in tvec:
    prob += b[t+1] == b[t] + x[t] - u[t]
    prob += b[t+1] >= bmin[t+1]
    prob += b[t] <= bmax
    prob += x[t] <= xmax*z[t]
            # Debugging tips: Du kan ikke constrainte en variabels startpunkt, når startpunktet har fået en startværdi.

# Export model
prob.writeLP("lp-files/mpc1.lp")

# Solve
status = prob.solve()

# Print results nicely
print("Status:", LpStatus[status])
print("Objective:", value(prob.objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Visualise results using plotly
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=tvec, y=[value(b[t]) for t in tvec], mode='lines', name='State-of-Charge'))
fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='Price'))
fig.update_xaxes(tickvals=tvec[::24], ticktext=[str(t//24) for t in tvec[::24]])
# add "Days" to x-axis
# Add total cost to title
fig.update_layout(
    title="MPC for Smart Charging of EVs (SIMULATED DATA)       Total cost: " + str(value(prob.objective)) + "DKK",
    xaxis_title="Days",
    yaxis_title="kWh or DKK/kWh",)
fig.show()

# Export interactive plotly figure
fig.write_html("../../plots/MPC/mpc1.html")

# Visualise results using plotly
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=tvec, y=[value(b[t]) for t in tvec], mode='lines', name='State-of-Charge'))
fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='Price'))
fig.update_xaxes(tickvals=tvec[::24], ticktext=[str(t//24) for t in tvec[::24]])
# add "Days" to x-axis
# Add total cost to title
fig.update_layout(
    title="MPC for Smart Charging of EVs (SIMULATED DATA)       Total cost: " + str(value(prob.objective)) + "DKK",
    xaxis_title="Days",
    yaxis_title="kWh or DKK/kWh",)
fig.show()