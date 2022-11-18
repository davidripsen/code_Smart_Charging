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
import seaborn as sns
sns.set_theme()
pd.set_option('display.max_rows', 500)
plot = True
use_carnot = True

# Read the csv files
df = pd.read_csv('data/forecastsGreenerEl/prognoser.csv', sep=',', header=0, parse_dates=True)
dfspot = pd.read_csv('data/spotprice/df_spot_since_sept22_HourDK.csv', sep=',', header=0, parse_dates=True)
dfc = pd.read_csv('data/forecastsCarnot/carnot_forecasts.csv', sep=',', header=0, parse_dates=True)


if use_carnot:
    #dfc = dfc[(dfc['SOURCE'] == 'carnot') & (dfc['COUNTRY_AREA_CODE'] == 'DK2')]
    dfc = dfc[(dfc['COUNTRY_AREA_CODE'] == 'DK2')]
    # Make new df with chosen columns
    df = pd.DataFrame({'Atime': dfc.CREATED_AT, 'Time': dfc.TIME_START, 'PredPrice': dfc.FORECAST_PRICE_KWH, 'TruePrice_Carnot': dfc.PRICE_KWH})



    df.Atime.value_counts().hist()
    plt.title('Distribution of forecasts lengths (DK2)')
    plt.show()
    Atimes = df.Atime.unique()

    # Convert Atime and Time to datetime
    df['Atime'] = pd.to_datetime(df['Atime'], format='%Y-%m-%d %H:%M:%S')
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')


    # Plot a timeseries for Atime[44] using plotly
    i = 23
    fig = px.line(df[df['Atime'] == Atimes[i]], x='Time', y='PredPrice', title='Carnot forecast for Atime number '+ str(i))
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

    for atime in Atimes[400:450]:
        # Print all forecasts for Atime
        print(df[df['Atime'] == atime])


    # Subset df for dev (DELETE!!!!!)
    df = df[:100]


    del dfc





if not use_carnot:
    # Convert from EUR/MWh to DKK/KWh
    eur_to_dkk = 7.44
    df['Price'] = eur_to_dkk * df['Price']/1000

    # Convert 'Price' to 'PredPrice'
    df['PredPrice'] = df['Price']
    df.drop(columns=['Price'], inplace=True)

# Convert Atime and Time to datetime
df['Atime'] = pd.to_datetime(df['Atime'], format='%Y-%m-%d %H:%M:%S')
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

# Append true spot price to the dataframe
dfspot = dfspot[dfspot['PriceArea'] == 'DK2']
dfspot.rename(columns={'HourUTC': 'Time', 'SpotPriceDKK':'TruePrice'}, inplace=True)
dfspot.drop(columns=['PriceArea', 'SpotPriceEUR'], inplace=True)
dfspot['TruePrice'] = dfspot['TruePrice']/1000 # Convert to kWh
dfspot['Time'] = pd.to_datetime(dfspot['Time'], format='%Y-%m-%d %H:%M:%S')

# (!) Merge df and dfspot on Time (!)
df = pd.merge(df, dfspot, on='Time', how='left')

# Delete rows in dfspot where dfspot.Time is not in df.Time, reverse order and reset index - and export
dfspot = dfspot[dfspot['Time'].isin(df['Time'])]
    # For the sake of Day-Ahead Smart Charge, ensure that dfspot is not much longer than df.Atime
endday = df.Atime.iloc[-1].date() + pd.Timedelta(days=2)
    # cut off dfspot at endday
dfspot = dfspot[dfspot['Time'] < pd.to_datetime(endday)]
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
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/RawPredictions_movie_CARNOT="+str(use_carnot)+".pdf")
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

# Plot beautiful histogram of prediction horizon and export plot
if plot:
    fig = px.histogram(df, x='Atime')
    fig.update_layout(title='Prediction horizon (h) for forecast made at different (A)times', xaxis_title='Atime', yaxis_title='Count')
    fig.show()
    fig.write_image("plots/PredictionHorizonsCarnot="+str(use_carnot)+".png")

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

# df with only known prices, for imput to Day-Ahead Smart Charge
dfk = pd.DataFrame(columns=['Atime'] + ['t' + str(i) for i in range(0,h+1)])
dfk['Atime'] = df['Atime'].unique()
wellknownhours = 48 - (dfk['Atime'].dt.hour + 1)
dfk['Atime_next'] = dfk['Atime'].shift(-1)
dfk['Atime_next'].iloc[-1] = endtime+pd.Timedelta(hours=1)
diff = pd.Series((pd.Series(dfk['Atime_next']).dt.ceil('H') - pd.Series(dfk['Atime']).dt.ceil('H'))).dt
dfk.insert(1, 'Atime_diff', (diff.days * 24 + diff.seconds/3600).astype(int))
dfk.drop(columns=['Atime_next'], inplace=True)
for j, wk in enumerate(wellknownhours):
    for i in range(0, wk):
        dfk.loc[j, 't' + str(i)] = dft.loc[j, 't' + str(i)]
dfk.fillna(BigM, inplace=True)

# Export to csv
dft.to_csv('data/MPC-ready/df_trueprices_for_mpc.csv', index=False)
dfp.to_csv('data/MPC-ready/df_predprices_for_mpc.csv', index=False)
dfk.to_csv('data/MPC-ready/df_knownprices_for_mpc.csv', index=False)

# For each Atime plot the Predicted Price (dfp) and TruePrice (dft) throughout the horizon
if not use_carnot:
    pdf = matplotlib.backends.backend_pdf.PdfPages("plots/ModPredictions_movie_Carnot="+str(use_carnot)+".pdf")
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