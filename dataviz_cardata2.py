""" 
    Read and pre-process 2nd upload of charge files and visualise EV user data from Monta
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import plotly.graph_objects as go
import datetime
pd.set_option('display.max_rows', 500)

dfA1 = pd.read_csv('data/Monta/charges_part1.csv', sep=',', header=0, parse_dates=True, low_memory=False)
dfA2 = pd.read_csv('data/Monta/charges_part2.csv', sep=',', header=0, parse_dates=True, low_memory=False)
dfA3 = pd.read_csv('data/Monta/charges_part3.csv', sep=',', header=0, parse_dates=True, low_memory=False)
D = pd.concat([dfA1, dfA2, dfA3], ignore_index=True)

# Sort feature to have these first in df: CABLE_PLUGGED_IN, RELEASED_AT, KWH, KWHS, PRICE, SPOT_PRICES
D = D[['CABLE_PLUGGED_IN', 'RELEASED_AT', 'KWH', 'KWHS', 'PRICE', 'SPOT_PRICES']]
# Rename columns
D.columns = ['CABLE_PLUGGED_IN', 'RELEASED_AT', 'KWH', 'KWHS', 'PRICE', 'SPOT_PRICES']




# Convert to datetime
timevars = ['CABLE_PLUGGED_IN_AT', 'RELEASED_AT', 'STARTING_AT', 'COMPLETED_AT', 'PLANNED_PICKUP_AT']
for var in timevars:
    D[var] = pd.to_datetime(D[var], format='%Y-%m-%d %H:%M:%S')



# Show$
D.info()
D.iloc[44]
# 1. Hmmm... Plug-in og plug-out are only dates, not times :-(, as before
# later: Calculate z_t = CABLE_PLUGGED_IN - RELEASED_AT (eller PLANNED_PICKUP_AT ?)

# Let's take a look at the data where SOC is available AND we are SMART CHARGING
D2 = D[D['SOC'].notna()]
D2 = D2[D2['SMART_CHARGE_ID'].notna()]
print("Disregarding ", round((1-(len(D2)/len(D))),4) *100, " % of the data where SOC is not available or we are not smart charging")
i = 2000

sesh = D2.iloc[i]
sesh

eval(sesh.KWHS)

sesh.KWHS
xt = pd.DataFrame(eval(sesh.KWHS))
prices = pd.DataFrame(eval(sesh.SPOT_PRICES))

# xt dot prices where time is the same
xt['time'] = pd.to_datetime(xt['time'])
prices['time'] = pd.to_datetime(prices['time'])
xt = xt.set_index('time')
prices = prices.set_index('time')

assert sesh.COST == round((xt * prices).sum()[0],4), "Costs do not match"
assert sesh.KWH == round(xt.sum()[0],4), "KWHs do not match"
assert sesh.SOC_START + sesh.KWH == sesh.SOC, "SOC_end does not match KWH charged + SOC_START"

print("SOC_START = ", sesh.SOC_START, "     SOC = ", sesh.SOC)

print("Number of different users:  ", D2.USER_ID.unique().shape[0])
print("Number of different vehicles:  ", D2.VEHICLE_ID.unique().shape[0])
print("Number of different smart chard IDs:  ", D2.SMART_CHARGE_ID.unique().shape[0])



############# Let's extract a single VEHICLE profile ########################################
vehicle_ids = D2.VEHICLE_ID.unique()
def PlotChargingProfile(D, var="VEHICLE_ID", id=22322):
    D2v = D[D[var] == id]
    D2v = D2v.sort_values(by=['CABLE_PLUGGED_IN_AT'])
    id = int(id)

    firsttime = D2v['CABLE_PLUGGED_IN_AT'].min()
    lasttime = D2v['RELEASED_AT'].max()

    fig = go.Figure(data=[go.Scatter(
        x=D2v['CABLE_PLUGGED_IN_AT'],
        y=D2v['KWH'],
        mode='lines+markers',
        name='Plug-in',
        marker=dict(
            size=10,
            opacity=0.8
        )
    )])

    fig.add_trace(go.Scatter(
        x=D2v['RELEASED_AT'],
        y=D2v['KWH'],
        mode='lines+markers',
        name = "Plug-out",
        marker=dict(
            size=10,
            opacity=0.8,
            color='purple'
        )
    ))
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
        yaxis_title_text="KWH charged in session", # yaxis label
        #font=dict(
        #    size=18,
        #    color="RebeccaPurple"
        #)
    )
    fig.show()

I = [3, 8, 10, 43, 1111, 500]
for i in I:
    PlotChargingProfile(D2, id=vehicle_ids[i])



#### Thoughts
# 1. VEHICLE_ID ≈ SMART_CHARGE_ID






############## What I need - how I need it  ###############################################
# z_t possible to calculate in near-future  :-)  z_t = CABLE_PLUGGED_IN - RELEASED_AT (eller PLANNED_PICKUP_AT ?), when Marcos has extracted as datetime
# u_t   = kan udregnes fra SOC (eller omvendt)
# b_t (SOC) = ???? (De få SOC-værdier, der er, ser ikke ud til at være korrekte)
#   - Det er vist ikke rigtig muligt at lave noget Multi-Day Smart Charge uden.
#   - De bruger den jo nok ikke, når de kun kigger frem til plug-out. Og de lader ekstra, hvis prisen er meget lav.
#   - Det er ikke faktisk ikke så tosset en idé bare at lade ekstra, når forecast prisen er høj de næste dage.
#   - Men er det egentlig ikke lidt det samme som implicit og ubevidst at forecaste u_t og z_t ?

# bmax (burde være der et sted) = KWH_LIMIT    (SOC_LIMIT = 80 or 100 % (presumably set by user))
# xmax = now_kwh (charge_smart_charges.csv) :-)

# Decision var (as they do it)
# xt = xt                                   :-)