"""
Read, process and plot the imbalance data.

Minor notes: Time is UTC and when the hour ENDED.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
pd.set_option('display.max_rows', 500)

D = pd.read_csv('data/PriceFor/TestReg.csv', parse_dates=True)
D['Time'] = pd.to_datetime(D['Time'], format='%Y-%m-%d %H:%M:%S')
D = D[D['Time'] >= '2022-06-01'].reset_index(drop=True)

# Discard all variables starting with status.
D = D[[x for x in D.columns if not x.startswith('status')]]
D = D[[x for x in D.columns if not x.startswith('pe_RegVol')]]

# Plot mean imbalance price.
fig = go.Figure()
fig.add_trace(go.Scatter(x=D['PTime'], y=D['pe_RegPrice.DK2'], name='RegPrice'))
fig.update_layout(title='RegPrice', xaxis_title='Time', yaxis_title='DKK/MWh')
fig.show()

# Plot all imbalance prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=D['Time'], y=D['pe_RegPrice.DK2'], name='pe_RegPrice.DK2'))
for col in D.columns[4:]:
    # Change opacity to see overlapping traces.
    fig.add_trace(go.Scatter(x=D['Time'], y=D[col], name=col, line=dict(width=0.5), opacity=0.8))
fig.update_layout(title='ImbalancePrice Scenarios', xaxis_title='Time', yaxis_title='DKK/MWh')
fig.show()

# Plot quantiles and vfill between them.
scenarios = D.iloc[:,4:-1]
quantiles = np.quantile(scenarios, [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[0,:], name='2.5%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[1,:], name='10%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[2,:], name='25%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[3,:], name='50%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[4,:], name='75%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[5,:], name='90%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=quantiles[6,:], name='97.5%', line=dict(width=1), opacity=0.8))
fig.add_trace(go.Scatter(x=D['Time'], y=D['pe_RegPrice.DK2'], name='RegPrice', line=dict(width=1.5), opacity=0.8))
fig.update_layout(title='ImbalancePrice Scenarios', xaxis_title='Time', yaxis_title='DKK/MWh')
fig.show()