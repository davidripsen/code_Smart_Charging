
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pd.read bigjobs_results.csv
D = pd.read_csv('results/bigjobs_results.csv')
#D = D.drop(columns=['Unnamed: 0'])
#D = D.iloc[0:31, :-1]
(D['obj_stochKM'] < D['obj_da']).mean()

# Relative performance as lambda function
RelativePerformance = lambda x, pf, dc: (pf-x)/(pf-dc)
AbsolutePerformance = lambda x, dc: dc-x

