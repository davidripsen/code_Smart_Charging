"""
Implementation of the economic MPC problem for multi-day Smart Charging of EVs on data from Monta.
"""

from pulp import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
if __name__ == '__main__':
    from FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge, ExtractEVdataForMPC, MultiDay, StochasticProgram, MultiDayStochastic, getMediods
else:
    from code_Smart_Charging.MPC.FunctionCollection import ImperfectForesight, PerfectForesight, plot_EMPC, DumbCharge, ExtractEVdataForMPC, MultiDay, StochasticProgram, MultiDayStochastic, getMediods
import os
now = datetime.now()
nowstring = now.strftime("%d-%m-%Y__%Hh_%Mm_%Ss")
pd.set_option('display.max_rows', 500)
np.random.seed(2812)

# Choose
runDeterministicReference = True
NOTE = 'BigBoi' # Optional message to output folder
print(NOTE)

# Save results, note and copy of code
os.mkdir('results/'+nowstring)
if NOTE != '':
    # Write note to file
    f = open('results/'+nowstring+'/NOTE.txt', 'w')
    f.write(NOTE)
    f.close()

    # Also make a copy of the code used
    os.mkdir('results/'+nowstring+'/code')
    os.system('cp code_Smart_Charging/MPC/mpc5_bigjobs.py results/'+nowstring+'/code/')
    os.system('cp code_Smart_Charging/MPC/FunctionCollection.py results/'+nowstring+'/code/')

# Metrics (with DumbCharge as baseline)
RelativePerformance = lambda x, pf, dc:   (pf-x)/(pf-dc)
AbsolutePerformance = lambda x, dc:       dc-x

# Models
models_h = ['stoch', 'mda'] #['stochKM', 'stoch', 'mda']
models_plain = ['da', 'pf', 'dc']
horizons = [3, 4, 5, 6]
models = models_plain + [models_h[i] + str(h) for i in range(len(models_h)) for h in horizons]

# n_clusters  (= n_scenarios)
n_clusters=20

# Read scenarios from txt
scenarios = np.loadtxt('./data/MPC-ready/scenarios.csv', delimiter=','); scenarios_all=scenarios;

# Load pickle file from data/MPC-ready
with open('data/MPC-ready/df_vehicle_list.pkl', 'rb') as f:
    DFV = pickle.load(f)

# Flip order of DFV list
DFV = DFV[::-1]

# Init bookkeeping of results add index from 0 to 99
results = pd.DataFrame(columns=[model for model in models]+ ['vehicle_id'], index=range(len(DFV)))
infeasibles = pd.DataFrame(columns=[model for model in models]+ ['vehicle_id'], index=range(len(DFV))).fillna(' - ')
relativePerformances = pd.DataFrame(columns=[model for model in models]+ ['vehicle_id'], index=range(len(DFV))).fillna(' - ')
absolutePerformances = pd.DataFrame(columns=[model for model in models]+ ['vehicle_id'], index=range(len(DFV))).fillna(' - ')

# Loop over vehicles
for i in range(len(DFV)):
    #i = 0 # i=2 Good performance (from stochastic model), i=3: Shitty performance
    dfv, dfspot, dfp, dft, timestamps, z, u, uhat, b0, r, bmin, bmax, xmax, c_tilde, vehicle_id, firsthour, starttime, endtime = ExtractEVdataForMPC(dfv=DFV[i], z_var='z_plan_everynight', u_var='use_lin',
                                                                                                                                                    uhat_var='use_org_rolling', bmin_var='SOCmin_everymorning', p=0.10) # (dfv=DFV[i], z_var='z_plan_everynight', u_var='use_lin',                                                                                                                                                                                                                   # uhat_var='use_org_rolling', bmin_var='SOCmin_everymorning', p=0.10)
    print('Vehicle ID = ', vehicle_id, ' (', i, '/', len(DFV)-1, ')')
    results['vehicle_id'][i] = vehicle_id
    infeasibles['vehicle_id'][i] = vehicle_id
    relativePerformances['vehicle_id'][i] = vehicle_id
    absolutePerformances['vehicle_id'][i] = vehicle_id
 
    #################################################### LET'S GO! ########################################################

    for h in horizons:
        # Stochastic (without kMediods)
        prob_stoch, x, b, flagFeasible_stoch = MultiDayStochastic(scenarios, n_clusters, dfp, dft, dfspot, u, uhat, z, h*24, b0, bmax, bmin, xmax, c_tilde, r, perfectForesight=False, maxh=6*24, KMweights=None)
        results['stoch'+str(h)][i] = round(prob_stoch['objective'],2)
        infeasibles['stoch'+str(h)][i] = '  ' if flagFeasible_stoch else ' x '
        plot_EMPC(prob_stoch, 'Stochastic Multi-Day Smart Charge (h = '+str(h)+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, BatteryCap=bmax, export_only=True, firsthour=firsthour, vehicle_id=vehicle_id)

        # # Stochastic with kMediods
        # mediods, weights = getMediods(scenarios_all, n_clusters=n_clusters)
        # prob_stochKM, x, b, flagFeasible_stochKM = MultiDayStochastic(mediods, n_clusters, dfp, dft, dfspot, u, uhat, z, h*24, b0, bmax, bmin, xmax, c_tilde, r, maxh=6*24, KMweights=weights)
        # results['stochKM'+str(h)][i] = round(prob_stochKM['objective'],2)
        # infeasibles['stochKM'+str(h)][i] = '  ' if flagFeasible_stochKM else ' x '
        # plot_EMPC(prob_stochKM, 'Stochastic Multi-Day (+kMediods) Smart Charge (h = '+str(h)+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, export_only=True, BatteryCap=bmax, firsthour=firsthour, vehicle_id=vehicle_id)

        if runDeterministicReference:
            ### Multi-Dayahead (Deterministic)
            # #h = 4*24 # 5 days horizon for the multi-day smart charge
            prob_mda, x, b, flagFeasible_mda = MultiDay(dfp, dft, dfspot, u, uhat, z, h*24, b0, bmax, bmin, xmax, c_tilde, r, maxh = 6*24, perfectForesight=False)
            results['mda'+str(h)][i] = round(prob_mda['objective'],2)
            infeasibles['mda'+str(h)][i] = '  ' if flagFeasible_mda else ' x '
            plot_EMPC(prob_mda, 'Multi-Day Smart Charge (h = '+str(h)+' days)  of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, export_only=True, BatteryCap=bmax, firsthour=firsthour, vehicle_id=vehicle_id)


    if runDeterministicReference:
        # Compare models on the data within horizon
        maxh = 6*24
        L = len(u) - (maxh+1)
        T = L - 1
        tvec = np.arange(T+1)
        T_within = T #-maxh 
        c_within = dfspot['TruePrice'][0:T_within+1] # Actually uses all prices in this case:-)
        tvec_within = tvec[0:T_within+1]
        z_within = z[0:T_within+1]
        u_within = u[0:T_within+1]
        u2_within = dfv['use'].to_numpy()[0:T_within+1]
        bmin_within = bmin[0:T_within+2]

        ### Day-Ahead SmartCharge
        prob_da, x, b, flagFeasible_da = MultiDay(dfp, dft, dfspot, u, uhat, z, 0, b0, bmax, bmin, xmax, c_tilde, r, DayAhead=True, maxh=6*24, perfectForesight=False)
        plot_EMPC(prob_da, 'Day-Ahead Smart Charge of vehicle = ' + str(vehicle_id), starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, export_only=True, BatteryCap=bmax, firsthour=firsthour, vehicle_id=vehicle_id)
        results['da'][i] = round(prob_da['objective'],2)
        infeasibles['da'][i] = '  ' if flagFeasible_da else ' x '

        ### Perfect Foresight
        prob_pf, x, b = PerfectForesight(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r, verbose=True)
        flagFeasible_pf = LpStatus[prob_pf.status] == 'Optimal'
        plot_EMPC(prob_pf, 'Perfect Foresight   of vehicle = ' + str(vehicle_id), x, b, u_within, c_within, z_within,  starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, export_only=True, BatteryCap=bmax, firsthour=firsthour, vehicle_id=vehicle_id)
        results['pf'][i] = round(value(prob_pf.objective),2)
        infeasibles['pf'][i] = '  ' if flagFeasible_pf else ' x '

        ### DumbCharge
        prob_dc, x, b = DumbCharge(b0, bmax, bmin_within, xmax, c_within, c_tilde, u_within, z_within, T_within, tvec_within, r=r, verbose=False)
        flagFeasible_dc = LpStatus[prob_dc.status] == 'Optimal'
        plot_EMPC(prob_dc, 'Dumb Charge   of vehicle = ' + str(vehicle_id) + '   r = '+str(r), x, b, u_within, c_within, z_within, starttime=str(starttime.date()), endtime=str(endtime.date()), export=True, export_only=True, BatteryCap=bmax, firsthour=firsthour, vehicle_id=vehicle_id)
        results['dc'][i] = round(value(prob_dc.objective),2)
        infeasibles['dc'][i] = '  ' if flagFeasible_dc else ' x '
        
        ### Evaluate performances
        for model in models:
            relativePerformances[model][i] = round(RelativePerformance(results[model][i], results['pf'][i], results['dc'][i]),2)
            absolutePerformances[model][i] = round(AbsolutePerformance(results[model][i], results['dc'][i]),2)
        relativePerformances['pf'][i] = -1*relativePerformances['pf'][i] # Reverse sign
        
    # Export results to file
    relativePerformances.to_csv('results/'+nowstring+'/relativePerformances.csv', index=False)
    results.to_csv('results/'+nowstring+'/results.csv', index=False)
    infeasibles.to_csv('results/'+nowstring+'/infeasibles.csv', index=False)
    absolutePerformances.to_csv('results/'+nowstring+'/absolutePerformances.csv', index=False)