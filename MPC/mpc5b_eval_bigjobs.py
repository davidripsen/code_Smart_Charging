
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
layout = dict(font=dict(family='Computer Modern',size=11),
              margin=dict(l=5, r=5, t=30, b=5),
              width=605, height= 250,
              title_x = 0.5)
path = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/Results/'
pathhtml = '/Users/davidipsen/Documents/DTU/5. Semester (MSc)/Thesis  -  SmartCharge/plots/_figures/'

# Specify
folder = '23-02-2023__12h_26m_42s'
manuel_title="Relative Total Cost of charging each random vehicle during test period"
nameofplot = 'randomtestresultsRTC'
measure = 'relativePerformances'
y_title = 'Relative Total Cost'

# Read data
D = pd.read_csv('results/'+folder+'/'+measure+'.csv')
D = D[D != ' - ']
#D = D.dropna()
D = D.astype(float)
order = ['pf', 'da'] + [i for i in D.columns if i[:5]=='stoch'] + [i for i in D.columns if i[:3]=='mda'] + ['dc'] # +['hist']

# Read txt file Note
with open('results/'+folder+'/NOTE.txt', 'r') as file:
    note = file.read()
Dres = pd.read_csv(f'results/'+folder+'/{measure}.csv')
Dres = Dres[Dres != ' - ']
#Dres = Dres.dropna()
Dres = Dres.astype(float)
Dres.columns = [col.replace('obj_','') for col in Dres.columns]
I = pd.read_csv('results/'+folder+'/infeasibles.csv')
Dfeas = D[I != ' x ']#.dropna() # Drop infeasible solutions

# Summary statisticso
round(D.describe(),2)
round(D.median(),2)
round(Dfeas.describe(),2)

#D['diff'] = D['stoch4'] - D['da']

# Make a boxplot of the values from each model using plotly. Give them distinct colors.
fig = go.Figure()
for i, col in enumerate(order): #enumerate(D.columns):
    if col not in ['vehicle_id']:
        fig.add_trace(go.Box(y=D[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=px.colors.qualitative.Plotly[i%10]))
fig.update_layout(title_text=f'Boxplot of {measure} of the models ('+note+')', title_x=0.5, showlegend=False)
if manuel_title: fig.update_layout(title_text=manuel_title)
fig.update_traces(boxmean=True)
fig.update_layout(xaxis_title_text='Model', yaxis_title_text = y_title)
if measure=='relativePerformances': fig.update_layout(yaxis_range=[-0.01, 1.01])
#else: fig.update_layout(yaxis_range=[-5, 1.02*D.max()['dc']])
fig.show()
fig.write_html(pathhtml+nameofplot+'.html')
fig.update_layout(layout)
fig.update_traces(line_width=1, marker_size=2)
fig.write_image(path+nameofplot+'.pdf')

RESULTS = pd.DataFrame(columns=['model','mean', 'median', 'stdofmean'])
for col in order:
        if col not in ['vehicle_id']:
            RESULTS.loc[len(RESULTS)] = [col+'_'+note, D[col].mean(), D[col].median(), D[col].std()/np.sqrt(len(D))]
RESULTS

# Repeat for only strictly feasible solutions
fig = go.Figure()
for i, col in enumerate(order):#enumerate(Dfeas.columns):
    if col not in ['vehicle_id']:
        fig.add_trace(go.Box(y=Dfeas[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=px.colors.qualitative.Plotly[i%10]))
fig.update_layout(title_text='Boxplot of {measure} of the models (only strictly feasible solutions) ('+note+')', title_x=0.5)
fig.update_traces(boxmean=True)
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

# Make histogram of the values from each model using plotly. Give them distinct colors.
fig = go.Figure()
for i, col in enumerate(D.columns):
    if col not in ['pf','dc','vehicle_id']:
        fig.add_trace(go.Histogram(x=D[col], name=col, marker_color=px.colors.qualitative.Plotly[i%10]))
fig.update_layout(title_text='Histogram of relative performances of the models', title_x=0.5, showlegend=False)
fig.update_traces(boxmean=True)
fig.show()


### Visualise the heatmap of I == 1' x '
I = I.replace(' x ', 1)
    # Move stoch4 and mda4 to the second and third column
I = I[['da','stoch3', 'mda3','pf','dc']]          # <----- adjust this bad boy to the models
fig = go.Figure(data=go.Heatmap(z=I, x=I.columns, y=I.index))
fig.update_layout(title_text='Heatmap of infeasible charge plans', title_x=0.5)
# Add xtext = 'Vehicle'
fig.update_traces(showscale=False)
fig.update_layout(xaxis_title_text='Model', yaxis_title_text='Vehicle')
fig.write_html(pathhtml+'infeasibles.html')
fig.update_layout(layout)
fig.write_image(path+'infeasibles.pdf')
fig.show()

(I.da== 1).mean()