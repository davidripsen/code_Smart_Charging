
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import scipy.stats as stats
sns.set_theme()
pio.renderers.default = "browser"
pd.set_option('display.max_rows', 200)

folder = '11-01-2023__22h_35m_10s'
#D = pd.read_csv('results/'+folder+'/relativePerformances.csv')
D = pd.read_csv('results/'+folder+'/relativePerformances.csv')
D = D[D != ' - ']
D = D.dropna()
D = D.astype(float)
Dres = pd.read_csv('results/'+folder+'/results.csv')
Dres = Dres[Dres != ' - ']
Dres = Dres.dropna()
Dres = Dres.astype(float)
Dres.columns = [col.replace('obj_','') for col in Dres.columns]

# Summary statistics
round(D.describe(),2)
round(D.median(),2)

# Make one plot of all the histograms of the values from each model using plotly. Give them distinct colors.
fig = go.Figure()
for i, col in enumerate(D.columns):
    if col not in ['pf','dc','vehicle_id']:
        fig.add_trace(go.Histogram(x=D[col], name=col, nbinsx=20, opacity=0.75, marker_color=px.colors.qualitative.Plotly[i%10]))
#fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(title_text='Histogram of relative performances of the models', title_x=0.5)
fig.show()

# Make a boxplot of the values from each model using plotly. Give them distinct colors.
fig = go.Figure()
for i, col in enumerate(D.columns):
    if col not in ['pf','dc','vehicle_id']:
        fig.add_trace(go.Box(y=D[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=px.colors.qualitative.Plotly[i%10]))
fig.update_layout(title_text='Boxplot of relative performances of the models', title_x=0.5)
fig.show()

# Calculate two-sample PAIRED t-test for each model pair and write nicely in an pandas dataframe
ttest = pd.DataFrame(columns=['model1','model2','t-statistic','p-value'])
for i, col1 in enumerate(Dres.columns):
    if col1 not in ['vehicle_id']:
        for j, col2 in enumerate(Dres.columns):
            if col2 not in ['vehicle_id']:
                if i<j:
                    ttest.loc[len(ttest)] = [col1, col2, stats.ttest_rel(Dres[col1], Dres[col2])[0], stats.ttest_rel(Dres[col1], Dres[col2])[1]]
round(ttest,3)
