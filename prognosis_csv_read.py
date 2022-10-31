 # Read prognoser.csv (as created by ../Spotprisprognose/data_extract.sh) using pandas
# and plot the prognosis for the next 24 hours.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import json
import plotly.express as px
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
plot = False

# Read the csv files
df = pd.read_csv('data/forecastsGreenerEl/prognoser.csv', sep=',', header=0, parse_dates=True)
dfspot = pd.read_csv('data/spotprice/df_spot.csv', sep=',', header=0, parse_dates=True)

# Convert from EUR/MWh to DKK/KWh
eur_to_dkk = 7.44
df['Price'] = eur_to_dkk * df['Price']/1000

# Convert Atime and Time to datetime
df['Atime'] = pd.to_datetime(df['Atime'], format='%Y-%m-%d %H:%M:%S')
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

# Convert 'Price' to 'PredPrice'
df['PredPrice'] = df['Price']
df.drop(columns=['Price'], inplace=True)

# Append true spot price to the dataframe
dfspot = dfspot[dfspot['PriceArea'] == 'DK2']
dfspot.rename(columns={'HourDK': 'Time', 'SpotPriceDKK':'TruePrice'}, inplace=True)
dfspot.drop(columns=['PriceArea', 'SpotPriceEUR'], inplace=True)
dfspot['TruePrice'] = dfspot['TruePrice']/1000 # Convert to kWh
dfspot['Time'] = pd.to_datetime(dfspot['Time'], format='%Y-%m-%d %H:%M:%S')

# (!) Merge df and dfspot on Time (!)
df = pd.merge(df, dfspot, on='Time', how='left')

# Delete rows in dfspot where dfspot.Time is not in df.Time, reverse order and reset index - and export
dfspot = dfspot[dfspot['Time'].isin(df['Time'])]
dfspot = dfspot.iloc[::-1]
dfspot.reset_index(drop=True, inplace=True)
dfspot.to_csv('data/spotprice/df_spot_commontime.csv', index=False)
endtime = dfspot['Time'].iloc[-1]
del dfspot


# Plot Price vs TruePrice using plotly
if plot:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['PredPrice'], mode='lines', name='PredPrice'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['TruePrice'], mode='lines', name='TruePrice'))
    fig.update_layout(title='Price vs TruePrice', xaxis_title='Time', yaxis_title='PredPrice')
    fig.show()

# For each unique Atime, plot the Price and TruePrice using matplotlib and save to pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/RawPredictions_movie.pdf")
if plot: # Change to run=True for plotting
    for Atime in df['Atime'].unique():
        dfA = df[df['Atime'] == Atime]
        fig = plt.figure()
        plt.plot(dfA['Time'], dfA['PredPrice'], label='PredPrice')
        plt.plot(dfA['Time'], dfA['TruePrice'], label='TruePrice')
        plt.title('Price vs TruePrice for Atime = ' + str(Atime))
        plt.xlabel('Time')
        plt.ylabel('PredPrice')
        plt.legend()
        #fig.savefig('plots/plot_' + str(Atime) + '.pdf')
        pdf.savefig(fig)
    pdf.close()


# For each unique Atime, print the Price and TruePrice
for Atime in df['Atime'].unique():
    dfA = df[df['Atime'] == Atime]
    print(dfA[['Atime', 'Time', 'PredPrice', 'TruePrice']])
    print(' ')

# Show prediction horizon at each Atime
print(df['Atime'].value_counts().to_string())
    # Hor = 101-200 hours (min. 4 day, consider 5 days and fill some steps)
minH = df['Atime'].value_counts().min()


##############################################################################
# Change df into MPC-friendly format with constant timesteps
##############################################################################
h = 200 # horizon: t = 0..h
BigM = int(25000)  # BigM padding value for padding forecasts [EUR/MWh]

def SliceDataFrame(df, h, var='PredPrice', use_known_prices=False, dftrue=None, BigM=20000):
    # Create dataframe with Price as values from df and where Time is split into columns from 0 to 100 and Atime as rows
    if var=='TruePrice': # Ensure for that residuals outside of horizon(forecasts) have a super high residual, not 0.
        BigM = BigM*2
    df2 = pd.DataFrame(columns=['Atime'] + ['t' + str(i) for i in range(0,h+1)])
    df2['Atime'] = df['Atime'].unique()
    for Atime in df['Atime'].unique():
        for i in range(0, h+1):
            vals = df[df['Atime'] == Atime][var].values
            # Pad vals until length h+1 with BigM
            vals = np.pad(vals, (0, np.max([0, h+1 - len(vals)+1]) ), 'constant', constant_values=BigM)
            df2.loc[df2['Atime'] == Atime, 't' + str(i)] = vals[i]

    # Calculate number of hours until next Atime
    df2['Atime_next'] = df2['Atime'].shift(-1)
    df2['Atime_next'] = df2['Atime_next'].fillna(df2['Atime'].iloc[-1])
    df2['Atime_next'].iloc[-1] = endtime+pd.Timedelta(hours=1)
    diff = pd.Series((pd.Series(df2['Atime_next']).dt.ceil('H') - pd.Series(df2['Atime']).dt.ceil('H'))).dt
    df2.insert(1, 'Atime_diff', (diff.days * 24 + diff.seconds/3600).astype(int))
    df2.drop(columns=['Atime_next'], inplace=True)

    if use_known_prices & (dftrue is not None):
        print('Using known prices')
        # Hours ahead where price is known
        wellknownhours = 48 - (df2['Atime'].dt.hour + 1)

        # Replace values        
        for j, wk in enumerate(wellknownhours):
            for i in range(0, wk):
                df2.loc[j, 't' + str(i)] = dftrue.loc[j, 't' + str(i)]

    return df2
dft = SliceDataFrame(df, h, var='TruePrice', BigM=BigM) #df with TruePrice as values
dfp = SliceDataFrame(df, h, var='PredPrice', use_known_prices=True, dftrue=dft, BigM=BigM) #df with (predicted) Price as values

# Export to csv
dft.to_csv('data/MPC-ready/df_trueprices_for_mpc.csv', index=False)
dfp.to_csv('data/MPC-ready/df_predprices_for_mpc.csv', index=False)

# For each Atime plot the Predicted Price (dfp) and TruePrice (dft) throughout the horizon
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/ModPredictions_movie.pdf")
if plot: # Change to run=True for plotting
    for i, Atime in enumerate(dfp['Atime']):
        fig = plt.figure()
        plt.plot(np.arange(0,h+1), dfp.iloc[i,2:(2+minH+1)], label='PredictedPrice')
        plt.plot(np.arange(0,h+1), dft.iloc[i,2:(2+minH+1)], label='TruePrice', linestyle='-.', color='black')
        plt.title('PredictedPrice vs TruePrice for Atime = ' + str(Atime))
        plt.xlabel('Time [h]')
        plt.ylabel('Price [EUR/MWh]')
        plt.legend()
        #fig.savefig('plots/PredMovie2/PredictedPrice_' + str(Atime) + '.pdf')
        pdf.savefig(fig)
    pdf.close()


##############################################################################  
    

# They are OK, but
# 1) When the first 12-36 hours are completely known, they should be part of the "forecast" (HANDLED)
#     - All forecasts are made AFTER that the spot prices are publicly available at NordPool.
#     - So it is reasonable to assume known prices for the rest of the day and the day after.
# 2) The forecasts (at Atime) typicaly starts with forecasting an hour or two of the alread passed time. (HANDLED)
# 3) Some days, multiple forecasts have been run. The forecasts do not agree (eventhough on known price). This should be dealt with.
#    Probably by using the latest. (HANDLED) if MPC is identifying latest Atime when run.