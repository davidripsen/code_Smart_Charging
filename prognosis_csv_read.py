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
plot_alot = True
use_carnot = True
layout = dict(font=dict(family='Computer Modern',size=11),
              margin=dict(l=5, r=5, t=30, b=5),
              width=605, height= 250)

# Read the csv files
#df = pd.read_csv('data/forecastsGreenerEl/prognoser.csv', sep=',', header=0, parse_dates=True)
dfspot = pd.read_csv('data/spotprice/df_spot_since_sept22_HourDK.csv', sep=',', header=0, parse_dates=True)
#dfc = pd.read_csv('data/forecastsCarnot/carnot_forecasts.csv', sep=',', head   er=0, parse_dates=True)
#dfc = pd.read_csv('data/forecastsCarnot/carnot_forecasts2.csv', sep=',', header=0, parse_dates=True)
dfc = pd.read_csv('data/forecastsCarnot/carnot_forecasts3.csv', sep=',', header=0, parse_dates=True)

if use_carnot:
    #dfc = dfc[(dfc['SOURCE'] == 'carnot') & (dfc['COUNTRY_AREA_CODE'] == 'DK2')]
    #dfc = dfc[(dfc['COUNTRY_AREA_CODE'] == 'DK2')]
    #df = pd.DataFrame({'Atime': dfc.CREATED_AT, 'Atime_org': dfc.CREATED_AT, 'Time': dfc.TIME_START, 'PredPrice': dfc.FORECAST_PRICE_KWH, 'TruePrice_Carnot': dfc.PRICE_KWH, 'Source': dfc.SOURCE})
    # Make new df with chosen columns
    dfc = dfc[(dfc['source'] == 'carnot')].reset_index(drop=True)
    df = pd.DataFrame({'Atime': dfc.created_at, 'Atime_org': dfc.created_at, 'Time': dfc.time_start, 'PredPrice': dfc.forecast_price_kwh, 'TruePrice_Carnot': dfc.price_kwh, 'Source': dfc.source})

    # Convert Atime and Time to datetime
    df['Atime'] = pd.to_datetime(df['Atime'], format='%Y-%m-%d %H:%M:%S')
    df['Atime_org'] = pd.to_datetime(df['Atime_org'], format='%Y-%m-%d %H:%M:%S')
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

    # Ceil Atime to the next 5 minutes  
    #df['Atimec'] = df['Atime'].dt.ceil('10min')

    # Round Atime to latest value where Atime changes a lot next time
    #df['Atime'] = df['Atime'].dt.round('10min')

    # Handle the fact that the forecasts doesn't have a unique timestamp for when they were fucking created (!)
    new_forecast = df.Atime_org.diff(1).dt.seconds > 120 # If Atime changes by more than 2 min, then it is a new forecast
    last_forecast = df[new_forecast == True].index -1
    print('Number of forecasts: ', len(df[new_forecast]))
    actual_Atimes = df.Atime_org[last_forecast]; print(actual_Atimes)

    # Set all Atimes to the last Atime before the new forecast
    j=-1
    for i in last_forecast:
        df.loc[j+1 : i, 'Atime'] = df.loc[i, 'Atime']
        j=i

    # Cut away nans
    df = df.dropna()

    # Cut away forecasts after 2022-11-11
    df = df[df['Atime'] < '2022-11-11']
    horizons = df.Atime.value_counts()
    if plot_alot:
        horizons.hist(bins=60)
        plt.title('Distribution of forecasts lengths')
        plt.show()

    # Cut away forecasts with less than 168 values
    min_horizon = 96
    print("Cutting away forecasts with less than 168 values, keeping ", (df.Atime.value_counts() >= min_horizon).mean()*100, "% of forecasts")
    df = df[df['Atime'].isin(horizons[horizons >= min_horizon].index)]

    # Is there big gaps in Atime?
    print("Is there big gaps in Atime?")
    Atime_diff = pd.Series(df.Atime.unique()).diff().dt.seconds
    # Plotly histogram of Atime_diff
    if plot_alot:
        fig = px.histogram(Atime_diff, x=Atime_diff, nbins=100)
        fig.show()
    Atimes = df.Atime.unique()
    [str(i) for i in Atimes]

    # Plot a timeseries for Atime[44] using plotly
    i = 78 # i=23
    if plot_alot:
        fig = px.line(df[df['Atime'] == Atimes[i]], x='Time', y='PredPrice', title='Carnot forecast for Atime number '+ str(i))
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()

    for i in [4, 78, 209, 700, 900, 1200, 1241]:
        print(df[df['Atime'] == Atimes[i]])
        # Fucking dublicated signal !
        # Forecasts (almost!)  always start with 22.00  == 24.00 in UTC+2 (summer time), 22 == 23 in UTC+1 (winter time)
        # Seems random when there is data directly from nordpool.


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


###### Insert known prices in Carnot forecasts (!)
    # Hours ahead where price is known  # Assume available at 13 o' clock CET/CEST
df['Atime_CET/CEST'] =  df.Atime.dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
df['DayAhead_avail'] = df['Atime_CET/CEST'].dt.hour >= 13
cnt = 0
for i, atime in enumerate(df.Atime.unique()):
    # atime = df.Atime.unique()[i]
    dfA = df[df['Atime'] == atime]
    # Cut away predictions BEFORE Atime
    dfA = dfA[dfA['Time'] >= dfA['Atime'].dt.floor('H')]

    # Assure right length
    #knownhours = pd.date_range(start=pd.Series(atime).dt.floor('H').min(), end=pd.Series(dfA['Time'].iloc[0]).min(), freq='1H')[:-1]
    atimeCOP = dfA['Atime_CET/CEST'].min()
    knownhoursCOP = pd.date_range(start = pd.Series(atimeCOP).dt.floor('H',ambiguous=False).min(),
                                  end = pd.Series(atimeCOP).dt.ceil('D',ambiguous=False).min() + (pd.Timedelta(hours=24) * int(dfA['DayAhead_avail'].unique()[0])),
                                  freq = '1H',
                                  tz = "Europe/Copenhagen",
                                  ambiguous="infer")[:-1]

    knownhoursCOPhours = knownhoursCOP.hour
    zeros = np.where(knownhoursCOPhours == 0)
    if sum(knownhoursCOPhours == 0) >= 2:
        knownhoursCOP = knownhoursCOP[:zeros[-1][-1]] # If there are two 0's, cut away after the last one.
        knownhoursCOPhours = knownhoursCOPhours[:zeros[-1][-1]]
        cnt=cnt+1
        print('Hours cut off because of two 0s at ', i, atime)

    zeros = np.where(knownhoursCOPhours == 0)

    # Check if two consecutive numbers are the same in knownhoursCOP
    timezone_change = 1 if sum(knownhoursCOPhours[1:] == knownhoursCOPhours[:-1]) >= 1 else 0
    if timezone_change:
        print('Timezone change at ', i, atime)

    if not dfA.DayAhead_avail.iloc[-1] and len(knownhoursCOP) > 0:
        if len(knownhoursCOPhours) > 24 - knownhoursCOPhours[0] + timezone_change:
            knownhoursCOP = knownhoursCOP[:zeros[-1][-1]] # If Days-Ahead is not AND there is more hours than up to 23.00, then cut away at the last 0.
            cnt=cnt+1
            print('Gap between Atime and First time is too long <=>  forecasts are missing ', i,' ', atime)

    assert len(knownhoursCOP) <= 24 + timezone_change + 24*dfA.DayAhead_avail.iloc[-1], "More than 24 hours of known prices, but not DayAhead_avail"
    # Concat knownhours to dfA
    knownhours = pd.Series(knownhoursCOP).dt.tz_convert("UTC").dt.tz_localize(None)
    dfA = pd.concat([pd.DataFrame({'Atime': atime, 'Atime_org': np.nan, 'Time': knownhours, 'PredPrice': np.nan, 'TruePrice_Carnot': np.nan, 'Source': 'nordpool_insert'}), dfA])
    dfA['l_hours_avail'] = len(knownhoursCOP)

    # Insert dfA in df
    df = df.drop(df[df['Atime'] == atime].index)
    df = pd.concat([df, dfA])
print('Number of times hours were cut off', cnt)

# (!) Merge df and dfspot on Time (!)
df = pd.merge(df, dfspot, on='Time', how='left')

# PredPrice = TruePrice if source=='nordpool_insert'
df.loc[df['Source'] == 'nordpool_insert', 'PredPrice'] = df['TruePrice']

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
if plot_alot:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['PredPrice'], mode='lines', name='PredPrice'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['TruePrice'], mode='lines', name='TruePrice'))
    fig.update_layout(title='Price vs TruePrice', xaxis_title='Time', yaxis_title='PredPrice')
    fig.show()

# For each unique Atime, plot the Price and TruePrice using matplotlib and save to pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/ModPredictions_movie_CARNOT="+str(use_carnot)+".pdf")
if plot: # Change to run=True for plotting
    for Atime in df['Atime'].unique():
        dfA = df[df['Atime'] == Atime]
        fig = plt.figure()
        plt.plot(dfA['Time'], dfA['PredPrice'], label='PredPrice')
        plt.plot(dfA['Time'], dfA['TruePrice'], label='TruePrice', linestyle='dashed')
        plt.axvline(x=Atime, color='r', linestyle='--', label='Atime of forecast')
        plt.title('Price vs TruePrice for Atime = ' + str(Atime))
        plt.xlabel('Time')
        plt.ylim([-0.1, df.PredPrice.max()])
        plt.ylabel('PredPrice')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        #fig.savefig('plots/Carnot/plot_' + str(Atime) + '.pdf')
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
    fig.write_image("plots/Carnot/PredictionHorizonsCarnot="+str(use_carnot)+".png")

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
    df2 = pd.DataFrame(columns=['Atime','l_hours_avail'] + ['t' + str(i) for i in range(0,h+1)])
    df2['Atime'] = df['Atime'].unique()
    for Atime in df['Atime'].unique():
        df2.loc[df2['Atime'] == Atime, 'l_hours_avail'] = int(df.loc[df['Atime'] == Atime, 'l_hours_avail'].iloc[0])
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

    # Greener El forecasts
    if use_known_prices & (dftrue is not None) & (not use_carnot):
        print('Using known prices')
        # Hours ahead where price is known  # Assume available at 13 o' clock CET/CEST
        wellknownhours = 48 - (df2['Atime'].dt.hour + 1)

        # Replace values        
        for j, wk in enumerate(wellknownhours):
            for i in range(0, wk):
                df2.loc[j, 't' + str(i)] = dftrue.loc[j, 't' + str(i)]
    return df2
dft = SliceDataFrame(df, h, var='TruePrice', BigM=BigM) #df with TruePrice as values
dfp = SliceDataFrame(df, h, var='PredPrice', use_known_prices=False, dftrue=dft, BigM=BigM) #df with (predicted) Price as values

if not use_carnot:
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
    dfk.to_csv('data/MPC-ready/df_knownprices_for_mpc.csv', index=False)

# Export to csv
dft.to_csv('data/MPC-ready/df_trueprices_for_mpc.csv', index=False)
dfp.to_csv('data/MPC-ready/df_predprices_for_mpc.csv', index=False)

# Import from csv
dft = pd.read_csv('data/MPC-ready/df_trueprices_for_mpc.csv')
dfp = pd.read_csv('data/MPC-ready/df_predprices_for_mpc.csv')

# For each Atime plot the Predicted Price (dfp) and TruePrice (dft) throughout the horizon
K_plots = len(dfp['Atime'].unique()) # 200
minH = df['Atime'].value_counts().min()
pdf = matplotlib.backends.backend_pdf.PdfPages("plots/Carnot/Sliced_Predictions_movie_Carnot="+str(use_carnot)+".pdf")
if plot: # Change to run=True for plotting
    for i, Atime in enumerate(dfp['Atime'][:K_plots]):
        fig = plt.figure(figsize=(6.3, 2.6))
        plt.plot(np.arange(0,minH+1), dfp.iloc[i,3:(3+minH+1)], label='Predicted price')
        plt.plot(np.arange(0,minH+1), dft.iloc[i,3:(3+minH+1)], label='True price', linestyle='--')
        plt.title('Predicted price vs True price')
        plt.xlabel('Time [h]')
        plt.ylabel('Price   [DKK/kWh]')
        plt.ylim([-0.1, df.PredPrice.max()])
        plt.grid(axis='x', linestyle='-')
        plt.xticks(np.arange(0, minH+1, 1.0))
        plt.axvline(x=Atime, color='r', linestyle='--', label='Actual time of forecast')
        plt.tight_layout()
        plt.legend(loc='upper right')
        # Change layout to the defined layout
        #fig.set_size_inches(6.3, 2.6)
        fig.tight_layout()
        # Change font
        plt.rcParams.update({'font.size': 11})
        plt.rcParams.update({'font.family': 'Computer Modern Roman'})

        plt.show()
        #fig.savefig('plots/PredMovie2/PredictedPrice_' + str(Atime) + '.pdf')
        pdf.savefig(fig)
    pdf.close()

##############################################################################  
    
#### About Greener El forecasts #####
# They are OK, but
# 1) When the first 12-36 hours are completely known, they should be part of the "forecast" (HANDLED)
#     - All forecasts are made AFTER that the spot prices are publicly available at NordPool.
#     - So it is reasonable to assume known prices for the rest of the day and the day after.
# 2) The forecasts (at Atime) typicaly starts with forecasting an hour or two of the alread passed time. (HANDLED)
# 3) Some days, multiple forecasts have been run. The forecasts do not agree (eventhough on known price). This should be dealt with.
#    Probably by using the latest. (HANDLED) if MPC is identifying latest Atime when run.