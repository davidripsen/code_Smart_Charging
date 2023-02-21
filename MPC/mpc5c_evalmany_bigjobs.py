
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

folders = ['15-02-2023__17h_06m_54s',
    '15-02-2023__17h_07m_06s',
    '15-02-2023__17h_06m_39s',
    '15-02-2023__17h_06m_25s',
    '15-02-2023__17h_06m_09s',
    '15-02-2023__17h_06m_00s',
    '15-02-2023__17h_05m_50s',
    '15-02-2023__17h_05m_42s',
    '15-02-2023__17h_05m_31s',
    '15-02-2023__17h_05m_23s',
    '15-02-2023__17h_05m_13s',
    '15-02-2023__17h_05m_04s',
    '15-02-2023__17h_04m_40s',
    '15-02-2023__17h_04m_31s',
    '15-02-2023__17h_04m_25s',
    '15-02-2023__17h_04m_19s',
    '16-02-2023__17h_23m_19s',
    '16-02-2023__17h_23m_39s',
    '16-02-2023__17h_24m_12s',
    '16-02-2023__17h_24m_49s',
    '16-02-2023__17h_25m_00s',
    '16-02-2023__17h_25m_10s',
    '16-02-2023__17h_25m_18s',
    '16-02-2023__17h_25m_25s',
    '20-02-2023__15h_41m_03s',
    '20-02-2023__15h_41m_19s',
    '20-02-2023__15h_41m_27s',
    '20-02-2023__15h_41m_36s',
    '20-02-2023__15h_41m_44s', # OPTIMAL
    '20-02-2023__15h_41m_54s',
    '20-02-2023__15h_42m_03s']

# Store mean and median of each model of each foldername/job
RESULTS = pd.DataFrame(columns=['model','mean', 'median', 'stdofmean'])

for folder in folders:
    folder = folder # folder = '20-02-2023__15h_41m_44s' (OPTIMAL)
    D = pd.read_csv('results/'+folder+'/results.csv')
    D = D[D != ' - ']
    #D = D.dropna()
    D = D.astype(float)
    # Read txt file Note
    with open('results/'+folder+'/NOTE.txt', 'r') as file:
        note = file.read()
        note = note[15:] # Remove some for readibility
    Dres = pd.read_csv('results/'+folder+'/results.csv')
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

    # Disregard index > 46
    D = D[D.index <= 46] # Special case

    # Store mean and median of each model across all jobs
    for col in D.columns:
        if col not in ['pf','dc','vehicle_id']:
            RESULTS.loc[len(RESULTS)] = [col+'_'+note, D[col].mean(), D[col].median(), D[col].std()/np.sqrt(len(D))]

    # Make a boxplot of the values from each model using plotly. Give them distinct colors.
    fig = go.Figure()
    for i, col in enumerate(D.columns):
        if col not in ['pf','dc','vehicle_id']:
            fig.add_trace(go.Box(y=D[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=px.colors.qualitative.Plotly[i%10]))
    fig.update_layout(title_text='Boxplot of relative performances of the models ('+note+')', title_x=0.5, showlegend=False)
    # Fix y-axis
    #fig.update_yaxes(range=[-5, 1050])
    fig.update_traces(boxmean=True)
    fig.show()
# fig.update_traces(boxmean=True)
# fig.write_html(pathhtml+'resultsBoxplot.html')
# fig.update_layout(layout)
# fig.update_traces(line_width=1, marker_size=2)
# fig.write_image(path+'resultsBoxplot.pdf')





# Repeat for only strictly feasible solutions
fig = go.Figure()
for i, col in enumerate(Dfeas.columns):
    if col not in ['pf','dc','vehicle_id', 'mda6']:
        fig.add_trace(go.Box(y=Dfeas[col], name=col, boxpoints='all', jitter=0.3, pointpos=-1.8, marker_color=px.colors.qualitative.Plotly[i%10]))
fig.update_layout(title_text='Boxplot of relative performances of the models (only strictly feasible solutions) ('+note+')', title_x=0.5)
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
I = I[['da','stoch3', 'stoch4', 'stoch5', 'stoch6', 'mda3','mda4','mda5','mda6',  'pf','dc']]          # <----- adjust this bad boy to the models
fig = go.Figure(data=go.Heatmap(z=I, x=I.columns, y=I.index))
fig.update_layout(title_text='Heatmap of infeasible charge plans', title_x=0.5)
# Add xtext = 'Vehicle'
fig.update_traces(showscale=False)
fig.update_layout(xaxis_title_text='Model', yaxis_title_text='Vehicle')
fig.write_html(pathhtml+'infeasibles.html')
fig.update_layout(layout)
fig.write_image(path+'infeasibles.pdf')
fig.show()

#(I.da== 1).mean()
#16 * 2 * 2*30*24 * 8/24 + 2*100 + 1 *2*30*24 * 8/24


# Analyse Grid Search RESULTS

# Which are the 5 best models?
RESULTS.sort_values(by='median', ascending=True).head(30)
    # Stoch: h = 3, lambda = 0.3, p=0.4
    # DA: p=0.3
    # MDA: h=3, lambda=0.2, p=0.4

    # N = 10

    # NOW: READ NEW FORECASTS