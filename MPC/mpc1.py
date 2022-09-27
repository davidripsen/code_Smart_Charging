"""
    Implementation of the economic MPC problem for multi-day Smart Charging of EVs.
    The problem is solved using the PuLP package.
"""
# Imports
from pulp import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt

# Read prices
df = pd.read_csv("../data/df_spot_2022.csv")
df['HourDK'] = pd.to_datetime(df['HourDK'])
    # Convert Spot prices to DKK/kWh
df['DKK'] = df['SpotPriceDKK']/1000
df = df.drop(['PriceArea', 'SpotPriceEUR', 'SpotPriceDKK'], axis=1)
    # Flip order of rows and reset index
df = df.iloc[::-1].reset_index(drop=True)

# Subset for approx 6 months
df = df[df['HourDK'] > '2022-03-28']
df.reset_index(drop=True, inplace=True)



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

def PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec, verbose=True):
    # Init problem
    prob = LpProblem("mpc1", LpMinimize)

    # Init variables
    x = LpVariable.dicts("x", tvec, lowBound=0, upBound=xmax, cat='Continuous')
    b = LpVariable.dicts("b", np.append(tvec,T+1), lowBound=0, upBound=bmax, cat='Continuous')
    b[0] = b0

    # Objective
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * ((b[T+1])-b[0]))

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t] - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t] <= bmax
        prob += x[t] <= xmax*z[t]
                # Debugging tips: Du kan ikke constrainte en variabels startpunkt, når startpunktet har fået en startværdi.

    # Solve problem
    if verbose:
        prob.solve()
    else:
       prob.solve(PULP_CBC_CMD(msg=0))

    # Return results
    return(prob, x, b)

# Run the problem
prob, x, b = PerfectForesight(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)

# Print results nicely
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Export model
prob.writeLP("MPC/lp-files/mpc1.lp")







######################################################
########### Visualise results using plotly ###########
######################################################

def plot_EMPC(prob, name="",export=False):
    # Identify iterative-appended, self-made prob
    fig = go.Figure()
    if type(prob) == dict:
        tvec = np.arange(0, len(prob['x']))
        tvec_b = np.arange(0, len(prob['b']))
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(prob['b'][t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['x'][t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['u'][t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(prob['c'][t]) for t in tvec], mode='lines', name='Price'))
        obj = prob['objective']
    else:
        tvec = np.arange(0, len(x))
        tvec_b = np.arange(0, len(b))
        obj = value(prob.objective)
        fig.add_trace(go.Scatter(x=tvec_b, y=[value(b[t]) for t in tvec_b], mode='lines', name='State-of-Charge'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(x[t]) for t in tvec], mode='lines', name='Charging'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(u[t]) for t in tvec], mode='lines', name='Use'))
        fig.add_trace(go.Scatter(x=tvec, y=[value(c[t]) for t in tvec], mode='lines', name='Price'))
    
    fig.update_xaxes(tickvals=tvec_b[::24], ticktext=[str(t//24) for t in tvec_b[::24]])
    # Fix y-axis to lie between 0 and 65
    fig.update_yaxes(range=[-3, 62])

    # add "Days" to x-axis
    # Add total cost to title
    fig.update_layout(
        title=name + "    MPC of EVs (simulated consumer data)        September 2022       Total cost: " + str(round(obj)) + " DKK",
        xaxis_title="Days",
        yaxis_title="kWh or DKK/kWh",)
    fig.show()

    ## Export figure
    if export:
        fig.write_html( "../plots/MPC/" + name + "_mpc.html")

# Plot results
plot_EMPC(prob, 'SmartCharge')




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
    prob += lpSum([c[t]*x[t] for t in tvec] - c_tilde * (b[T+1]-b[0]))

    # Constraints
    for t in tvec:
        prob += b[t+1] == b[t] + x[t] - u[t]
        prob += b[t+1] >= bmin[t+1]
        prob += b[t] <= bmax
        
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

# Run the problem
prob, x, b = DumbCharge(b0, bmax, bmin, xmax, c, c_tilde, u, z, T, tvec)
print("Status:", LpStatus[prob.status])

# Plot results
plot_EMPC(prob, 'DumbCharge')

# Get delta difference of HourDK
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))

df[2038:2045]

px.line(np.diff(df.HourDK)).show()





######################################################
### 3. Implementation of DAY-AHEAD Smart Charge ######
######################################################

c = df['DKK'].to_numpy()
T = len(c)-1
tvec = np.arange(T+1)

# External variables (SIMULATED)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
c_tilde = min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h
u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])

def DayAhead(dff, b0, bmax, bmin, xmax, c_tilde, u, z, tvec):
    # Identify plug-ins: When z turns from 0 to 1
    indx_plug_ins = np.where(np.diff(z) == 1)[0]
    
    # Do we start with the car plugged in?
    if z[0] == 1:
        indx_plug_ins = np.append(0, indx_plug_ins)
    
    # Init result store
    xs = []
    bs = []; bs.append([b0])
    us = []
    cs = []

    for it, plug_in in enumerate(indx_plug_ins):
        time_plug_in = dff['HourDK'][plug_in]
        # Identift next plug_in
        plug_in_next = indx_plug_ins[it+1] if it < len(indx_plug_ins)-1 else dff.index[len(dff)-1]+1

        if time_plug_in.hour >= 13:  ### ASSUME day-ahead prices are available from 13 o'clock
            avail_day_ahead = True
        else:
            avail_day_ahead = False

        # Define available day-ahead prices
        if avail_day_ahead:
            last_price_avail = time_plug_in + dt.timedelta(days=1)
            last_price_avail = last_price_avail.replace(hour=23)
        else:
            last_price_avail = time_plug_in.replace(hour=23)

        # Subset the day-ahead problem
        flex_indx = ((dff['HourDK'] >= time_plug_in) & (dff['HourDK'] <= last_price_avail)).to_numpy()
        c_flex = dff['DKK'][flex_indx].to_numpy()
        c_tilde = min(c_flex)
        h = len(c_flex)-1
        tvec_flex = np.arange(0,h+1)

        # Find relevant input at the specific hours of flexibility
        z_flex = z[flex_indx]
        u_flex = u[flex_indx]
        #bmin_flex = np.piecewise(np.append(tvec_flex,T+1), [np.append(tvec_flex,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])
        bmin_flex = bmin[np.append(tvec_flex, tvec_flex[-1]+1)]

        # For known day-ahead prices, this is a perfect foresight problem
        prob, x, b = PerfectForesight(b0, bmax, bmin_flex, xmax, c_flex, c_tilde, u_flex, z_flex, h, tvec_flex)
        b0 = value(b[h+1]) # Save b0 for next iteration

        # Store results
        xs.append([value(x[t]) for t in np.arange(0, plug_in_next - plug_in)])
        bs.append([value(b[t]) for t in np.arange(0, plug_in_next - plug_in)])
        us.append([value(u_flex[t]) for t in np.arange(0, plug_in_next - plug_in)])
        cs.append([value(c_flex[t]) for t in np.arange(0, plug_in_next - plug_in)])

    # Flatten list of lists
    xss = [item for sublist in xs for item in sublist]
    bss = [item for sublist in bs for item in sublist]
    uss = [item for sublist in us for item in sublist]
    css = [item for sublist in cs for item in sublist]

    # Costs
    total_cost = np.dot(np.array(xss), np.array(css)) - c_tilde * (bss[-1]-bss[0])

    # Tie results intro prob
    prob = {'x':xss, 'b':bss, 'u':uss, 'c':css, 'objective':total_cost}
    return(prob, x, b)

# Run the problem
prob, x, b = DayAhead(df, b0, bmax, bmin, xmax, c_tilde, u, z, tvec)
plot_EMPC(prob, 'DayAhead')



#####################################################
### 4. Multi-day Smart Charging #####################
#####################################################
####### Designed for re-run every hour ##############

# External variables (SIMULATED) (SLAP KODE LIGE HER)
T = 4*24 # 4 days
N = len(df)
T = N # Length of experiment
tvec = np.arange(T)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
u = np.random.uniform(8,16,T) * (tvec % 24 == np.floor(plugin)-1)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])

# Horizon of multi-day
h = 4*24 # 4 days

def MultiDay(df, u, z, h, b0, bmax, bmin, xmax, c_tilde):
    N = len(df)
    L = N-h # Length of experiment
    tvec = np.arange(0,h+1)
    B = np.empty((L+1)); B[:] = np.nan; B[0] = b0;
    X = np.empty((L)); X[:] = np.nan

    # Loop over all hours, where there is still T hours remaining of the data
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
    






######################################################
### 5. Run models on real prices #####################
######################################################

h = 4*24 # 4 days horizon for the multi-day smart charge
c = df['DKK'].to_numpy()
c_tilde = np.quantile(c, 0.1) #min(c[-0:24]) # Value of remaining electricity: lowest el price the past 24h

# External variables (SIMULATED) 
T = len(c)-1# -1-h # N-1 # N-T
tvec = np.arange(T+1)
z = np.piecewise(tvec, [(((tvec % 24) >= np.ceil(plugin)) | ((tvec % 24) <= np.floor(plugout-0.01)))], [1,0]) # [0,1] plugged in at tc = 5.5 - z*np.random.uniform(-1,2,T+1) # cost of electricity at t
u = np.random.uniform(8,16,T+1) * (tvec % 24 == np.floor(plugin)-1) # uniform(8,16, T eller T+1? MANGLER)
bmin = np.piecewise(np.append(tvec,T+1), [np.append(tvec,T+1) % 24 == np.ceil(plugout)], [bmin_morning, 1])

# Compare models on the data within horizon
T_within = T - h
c_within = c[0:T_within+1]
tvec_within = tvec[0:T_within+1]
z_within = z[0:T_within+1]
u_within = u[0:T_within+1]
bmin_within = bmin[0:T_within+2]
df_within = df[0:T_within+1]

##########################
### Perfect Foresight
prob, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within)
plot_EMPC(prob, 'Perfect Foresight', export=True)

### Dumb Charge
prob, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within)
plot_EMPC(prob, 'Dumb Charge', export=True)
# Juhuu :-D  Næsten 5 gange besparelse!!!
# MAGLER: Fix at det kun virker nogle gange - noget med endpoints for z?

### DayAhead
# Ensure to evaluate on the same days as the other models
# df.iloc[:-h+1]
prob, x, b = DayAhead(df_within, b0, bmax, bmin_within, xmax, c_tilde, u_within, z_within, tvec_within)
plot_EMPC(prob, 'Day-ahead Smart Charge', export=True)

### MultiDay
prob, x, b = MultiDay(df, u, z, h, b0, bmax, bmin, xmax, c_tilde)
plot_EMPC(prob, 'Multi-Day (Perfect Foresight) Smart Charge', export=True)
    # Perfect Foresight = MultiDay(h=inf)
    # Dumb Charge = MultiDay(h=0)
    # DayAhead = MultiDay(h=[12-36])