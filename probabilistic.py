"""
Start implementing the probabilistic regression add-on to the point estimate forecasts
In order to later generate scenarios for stochastic optimization
"""

# Imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt

# Read the dfp and dft
dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv', sep=',', header=0, parse_dates=True)

dft['Atime'] = pd.to_datetime(dft['Atime'], format='%Y-%m-%d %H:%M:%S')
dfp['Atime'] = pd.to_datetime(dfp['Atime'], format='%Y-%m-%d %H:%M:%S')

# Calculate df for residuals
dfr = dft - dfp
dfr['Atime'] = dfp['Atime']
dfr['Atime_diff'] = dfp['Atime_diff']

# Make beautiful matplotlib histograms of the residuals for each time step and save them all in one pdf
steps = dfr.columns[2:]

pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Histograms_of_residuals_per_timestep.pdf")
for step in steps:
    fig = plt.figure()
    plt.hist(dfr[step], bins=50)
    plt.title('Histogram of residuals for timestep ' + str(step))
    plt.xlabel('Residual')
    plt.ylabel('Count')
    pdf.savefig(fig)
pdf.close()