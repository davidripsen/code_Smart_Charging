"""
    Implementation of the economic MPC problem for multi-day Smart Charging of EVs.
    The problem is solved using the PuLP package.
"""
# Imports
from re import X
from pulp import *
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Read prices
df = pd.read_csv("../data/df_spot_month.csv")
df['HourDK'] = pd.to_datetime(df['HourDK'])
    # Convert Spot prices to DKK/kWh
df['DKK'] = df['SpotPriceDKK']/1000
df = df.drop(['PriceArea', 'SpotPriceEUR', 'SpotPriceDKK'], axis=1)
    # Flip order of rows and reset index
df = df.iloc[::-1].reset_index(drop=True)
    # Subset the first three days of df
dfsub = df.iloc[:72]

############# SIMULATION OF DATA: EV behaviour and prices #################
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


###################################################################
################## IMPLEMENTATION OF THE PROBLEM ##################
###################################################################

def PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec):
    # Init problem
    prob = LpProblem("mpc1", LpMinimize)

    # Init variables
    global x
    global b
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
    b[0] = b0

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * b[T+1])

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t] - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t] <= bmax
        prob += x[t] <= xmax*z[t]
                # Debugging tips: Du kan ikke constrainte en variabels startpunkt, når startpunktet har fået en startværdi.

    # Solve problem
    prob.solve()

    # Return results
    return(prob)

# Run the problem
prob = PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)

# Print results nicely
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Export model
prob.writeLP("MPC/lp-files/mpc1.lp")


######################################################
########### Visualise results using plotly ###########
def plot_EMPC(prob, tvec, name="", timestamps="",export=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tvec, y=[value(b[t]) for t in tvec], mode='lines', name='State-of-Charge'))
    fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
    fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
    fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='Price'))
    fig.update_xaxes(tickvals=tvec[::24], ticktext=[str(t//24) for t in tvec[::24]])
    # add "Days" to x-axis
    # Add total cost to title
    fig.update_layout(
        title=name + "    MPC of EVs (simulated data)       Total cost: " + str(round(value(prob.objective))) + " DKK",
        xaxis_title="Days",
        yaxis_title="kWh or DKK/kWh",)
    fig.show()

    ## Export figure
    if export:
        fig.write_html("../../plots/MPC/mpc1.html")

# Plot results
plot_EMPC(prob, tvec, 'SmartCharge')




######################################################
### 2. Implementation of INSTANT/DUMB CHARGE
######################################################

def DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec):
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
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * b[T+1])

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t] - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t] <= bmax
        
        ######## DUMB CHARGE ######## (MISSING: working)
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
    return(prob)

# Run the problem
prob = DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)

# Plot results
plot_EMPC(prob, tvec, 'DumbCharge')





######################################################
### 3. Implementation of day-ahead
######################################################
def DayAhead(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec):
    # Identify plug-ins: When z turns from 0 to 1
    indx_plug_in = np.where(np.diff(z) == 1)[0]
    #times_plug_in = df['HourDK'][indx_plug_in]


    for i in indx_plug_in:
        time_plug_in = df['HourDK'][i]
        if time_plug_in.hour >= 15:  ### ASSUME day-ahead prices are available from 15.00 (SIMULATION)
            avail_day_ahead = True
        else:
            avail_day_ahead = False

        if avail_day_ahead:
            # Sample timestamps from time_plug_in to time_plug_in + the whole next day
            # MISSING: NÅET HERTIL :-)








######################################################
### 4. Run models on real prices #####################
######################################################

c = df['DKK'].to_numpy()
T = len(c)-1
tvec = np.arange(0,T+1)

# External variables (SIMULATED)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
c_tilde = min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h
u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])

#######
# Perfect Foresight
prob = PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)
plot_EMPC(prob, tvec, 'PerfectForesight')

# Dumb Charge
prob = DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)
plot_EMPC(prob, tvec, 'DumbCharge')
# Juhuu :-D Besparelse: ca. 66 % af prisen.