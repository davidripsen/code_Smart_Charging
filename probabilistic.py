"""
Start implementing the probabilistic regression add-on to the point estimate forecasts
In order to later generate scenarios for stochastic optimization
"""

# Imports
import numpy as np
from sklearn.covariance import ShrunkCovariance
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt
import seaborn as sns

path = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/Carnot'
pathhtml = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/_figures'
zeromean = False
plot=True
# Plotly layout
layout = dict(font=dict(family='Computer Modern',size=11),
              margin=dict(l=5, r=5, t=30, b=5),
              width=605, height= 250,
              title_x = 0.5)

# Matplotlib layout
plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Computer Modern Serif', 'font.size': 11, 'figure.figsize': (6.3, 2.6), 'text.usetex': True})


# Read the dfp and dft
dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv', sep=',', header=0, parse_dates=True)
dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv', sep=',', header=0, parse_dates=True)

dft['Atime'] = pd.to_datetime(dft['Atime'], format='%Y-%m-%d %H:%M:%S')
dfp['Atime'] = pd.to_datetime(dfp['Atime'], format='%Y-%m-%d %H:%M:%S')

# Calculate df for residuals
dfr = dft - dfp
dfr['Atime'] = dfp['Atime']
dfr['Atime_diff'] = dfp['Atime_diff']

# Replace BigM with nan
BigM = dfr.max().iloc[-1] # 25000
dfr = dfr.replace(BigM, np.nan)

# Make beautiful matplotlib histograms of the residuals for each time step and save them all in one pdf
maxstep = 145
steps = dfr.columns[3:(3+maxstep)]

if plot:
    pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Carnot/Histograms_of_residuals_per_timestep_Carnot.pdf")
    for step in steps:
        fig = plt.figure()
        plt.hist(dfr[step][~np.isnan(dfr[step])], bins=50, density=True)
        nans = np.isnan(dfr[step])
        plt.title('Histogram of residuals for timestep '+ str(step) +'         (nans:'+ str(np.sum(nans))+' / '+str(len(dfr))+')')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # Plot Gaussian fit of the histogram data
        mu, std = np.mean(dfr[step][~np.isnan(dfr[step])]), np.std(dfr[step][~np.isnan(dfr[step])])
        if std != 0:
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * std**2) )
            plt.plot(x, p, 'k', linewidth=1, linestyle='--')

        plt.xlabel('Residual')
        plt.ylabel('Density')
        plt.xlim([-5,5])
        plt.ylim([0,0.65])
        #plt.show()
        pdf.savefig(fig)
    pdf.close()

### Fit multivariate normal distribution to the residuals

# ##### Calculate mean and covariance matrix
# from sklearn.covariance import ShrunkCovariance
# from sklearn.covariance import LedoitWolf
# from sklearn.covariance import OAS
# from sklearn.covariance import MinCovDet
# from sklearn.covariance import GraphicalLasso
# from sklearn.covariance import GraphicalLassoCV
# from sklearn.covariance import EmpiricalCovariance
# from sklearn.covariance import EllipticEnvelope
# from sklearn.covariance import ledoit_wolf
# from sklearn.covariance import oas
# from sklearn.covariance import empirical_covariance
# from sklearn.covariance import shrunk_covariance
# from sklearn.covariance import min_cov_det
# from sklearn.covariance import graphical_lasso
# from sklearn.covariance import graphical_lasso_path
# from sklearn.covariance import graphical_lasso_cv

# # Calculate mean and covariance matrix
# cov = MinCovDet().fit(df)
# cov = GraphicalLassoCV().fit(df)
# cov = GraphicalLasso().fit(df)
# cov = LedoitWolf().fit(df)
# cov = OAS().fit(df)
# cov = ShrunkCovariance().fit(df)
# cov = EmpiricalCovariance().fit(df)
# cov = EllipticEnvelope().fit(df)
# cov = ledoit_wolf(df)
# cov = oas(df)
# cov = empirical_covariance(df)
# cov = shrunk_covariance(df)
# cov = min_cov_det(df)
# cov = graphical_lasso(df)
# cov = graphical_lasso_path(df)
# cov = graphical_lasso_cv(df)

# In the small-samples situation, in which n_samples is on the order of n_features or smaller, sparse inverse covariance estimators tend to work better than shrunk covariance estimators.
# However, in the opposite situation, or for very correlated data, they can be numerically unstable. In addition, unlike shrinkage estimators, sparse estimators are able to recover off-diagonal structure.
# https://scikit-learn.org/stable/modules/covariance.html

# Drop all columns with nan
df = dfr.iloc[:,3:maxstep+3].to_numpy()
mu = dfr.iloc[:,3:maxstep+3].mean(numeric_only=True)
if zeromean:
    mu[0:maxstep] = 0
#cov = ShrunkCovariance(shrinkage=0.1).fit(df).covariance_
cov = np.cov(df, rowvar=False)
cov = dfr.iloc[:,3:maxstep+3].cov()
    # "the sample covariance matrix was singular which can happen from exactly collinearity (as you've said) or when the number of observations is less than the number of variables."

# Shrinking the covariance matrix
def ShrinkCov(alpha, cov):
    return (1-alpha) * cov + alpha * np.diag(np.diag(cov))
    # alpha = 0: no shrinkage
    # alpha = 1: shrinkage to the identity matrix

# Visualise mu (and therefore bias)
if plot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mu.index, y=mu.values, mode='lines+markers', name='Mean'))
    fig.update_layout(title='Mean of the residuals per timestep', xaxis_title='Timestep', yaxis_title='Mean')
    # Center title
    fig.update_layout(showlegend=False)
    fig.write_html(pathhtml + "/Mean_of_residuals_per_timestep_Carnot.html")
    fig.show()
    fig.update_layout(layout)
    fig.write_image(path + "/Mean_of_residuals_per_timestep_Carnot.pdf")
    #

# Make the above plots into a 2 x 1 plot
from plotly.subplots import make_subplots
Sigma = ShrinkCov(0, cov)
Sigma_shrink = ShrinkCov(0.3, cov)
fig = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.1, shared_yaxes=True)
fig.add_trace(go.Heatmap(
            z=Sigma,
            x=np.arange(maxstep),#cov.columns,
            y=np.arange(maxstep),#cov.columns,
            colorscale='Viridis',zmin=0,zmax=2), row=1, col=1)
fig.update_layout(showlegend=False)
fig.add_trace(go.Heatmap(
            z=Sigma_shrink,
            x=np.arange(maxstep),#cov.columns,
            y=np.arange(maxstep),#cov.columns,
            colorscale='Viridis',zmin=0,zmax=2), row=1, col=2)
# Set legend range plotly
fig.update_layout(title='Sample covariance matrix of residuals (left) and shrinked covariance matrix (right)')
fig.update_xaxes(title_text="Timestep", row=2, col=1)
fig.update_yaxes(title_text="Timestep", row=1, col=1)
fig.write_html(pathhtml + "/Covariance_matrix_of_residuals_Carnot.html")
fig.update_layout(layout)
fig.write_image(path + "/Covariance_matrix_of_residuals_Carnot.pdf")
fig.show()

alphas = [0.10, 0.20, 0.30, 0.40]
for alpha in alphas:
    COV=ShrinkCov(alpha, cov)    
    # Generate 100 samples from the multivariate normal distribution
    samples = np.random.multivariate_normal(mu.to_numpy(), COV.to_numpy(), 20000)
    print(samples.shape)
    # Export samples to csv
    np.savetxt(f"./data/MPC-ready/scenarios_shrunk_alpha={alpha}.csv", samples, delimiter=",")


# Visualise the time series of the samples and add 95 % prediction interval
if plot:
    samples = np.random.multivariate_normal(mu.to_numpy(), COV.to_numpy(), 20000)
    fig = go.Figure()
    # Add theoretical mean
    # Calculate emperical 95 % quantiles from samples
    #quantiles = np.quantile(samples, [0.025, 0.975], axis=0)
    #fig.add_trace(go.Scatter(x=mu.index, y=quantiles[0,:], mode='lines', name='97.5% quantile', line=dict(width=1, color='red', dash='dash')))
    #fig.add_trace(go.Scatter(x=mu.index, y=quantiles[1,:], mode='lines', name='2.5% quantile', line=dict(width=1, color='red', dash='dash')))
    for i in range(0,10):
        fig.add_trace(go.Scatter
            (x=mu.index, y=samples[i,:], mode='lines', name="Sample "+str(i), line=dict(width=1)))
    fig.update_layout(title='Statistical scenarios sampled from the multivariate normal distribution', xaxis_title='Timestep', yaxis_title='Residual')
    # Add 95% confidence interval
    fig.add_trace(go.Scatter(x=mu.index, y=mu.values+1.96*np.sqrt(np.diag(cov)), mode='lines', name='97.5% CI (Wald)', line=dict(width=1, color='red')))
    fig.add_trace(go.Scatter(x=mu.index, y=mu.values-1.96*np.sqrt(np.diag(cov)), mode='lines', name='2.5% CI (Wald)', line=dict(width=1, color='red')))
    fig.add_trace(go.Scatter(x=mu.index, y=mu.values, mode='lines', name='Mean', line=dict(color='black', width=1)))
    # Change legend to be horizontal below the plot
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-.22,
        xanchor="right",
        x=1
    ))
    fig.update_yaxes(range=[-3.2, 3.2])
    # Center title
    fig.write_html(pathhtml + "/Samples_from_multivariate_normal_distribution_Carnot.html")
    fig.show()
    fig.update_layout(layout)
    fig.update_layout(showlegend=False)
    fig.write_image(path + "/Samples_from_multivariate_normal_distribution_Carnot.pdf")

    # = Prediction interval (under correct model assumption)


# Calculate autocorrelation and partial autocorrelation in dfr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plot_acf(dfr['t48'], lags=145).show()
plot_pacf(dfr['t48'], lags=145).show()
    # There is ofc autocorrelation, that is what we are modelling with the covariance matrix


######################################################################################################################################
#################################################### FURTHER RESIDUAL ANALYSIS #######################################################
######################################################################################################################################
# Set seed
np.random.seed(2812)

fig = px.line(dfr, x='Atime', y='t48', title="Residuals at timestep 48")
fig.show()

for t in [24, 48, 72, 96, 120, 144]:
    # Make boxplots of residuals at t48 for each hour of the day as extracted from Atime
    dfr['hour'] = dfr['Atime'].dt.hour
    fig = px.box(dfr, x='hour', y='t'+str(t), title="Boxplots of residuals at timestep "+str(t) + " for each hour of the day")
    fig.update_yaxes(title_text="Residuals", range=[-4, 4])
    fig.show()


scenarios_all = samples
for k in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
    n_clusters = 10
    c_forecast = dfp.iloc[k, 3:3+145].to_numpy()
    c_true = dft.iloc[k, 3:3+145].to_numpy()
    c_s = dfp.iloc[k, 3:3+145].to_numpy() + scenarios_all
    c_s[c_s<0] = 0

    # Visualise n scenarios
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(c_forecast)), y=c_forecast, name='Forecast'))
    fig.add_trace(go.Scatter(x=np.arange(len(c_true)), y=c_true, name='True'))
    for i in range(n_clusters):
        fig.add_trace(go.Scatter(x=np.arange(len(scenarios_all[i])), y=c_s[i], name='Scenario '+str(i)))
    fig.update_layout(title=str(n_clusters) + ' Scenarios  at k='+str(k), xaxis_title='Time', yaxis_title='Price')
    fig.update_yaxes(range=[-1, 6])
    fig.show()