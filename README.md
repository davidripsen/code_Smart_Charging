# Smart Charging Electric Vehicles
Welcome!
In this repository you will find all the code and models for the Masters Thesis of David Ribberholt Ipsen.
Checkout [visualisations](https://davidripsen.github.io/code_Smart_Charging/) to see some of the results.

## Code
The most important implementations is located in `MPC/`, particularly `MPC/FunctionCollection.py`.

## Figures
[1](https://davidripsen.github.io/code_Smart_Charging/_figures/spotprices)
[2](https://davidripsen.github.io/code_Smart_Charging/_figures/infeasibles.html)
[3](https://davidripsen.github.io/code_Smart_Charging/_figures/assumptions_id10885.html)
[4](https://davidripsen.github.io/code_Smart_Charging/_figures/Day-Ahead%20Smart%20Charge%20of%20vehicle%20=%20853_mpc.html)
[5](https://davidripsen.github.io/code_Smart_Charging/_figures/Dumb%20Charge%20of%20vehicle%20=%20853%20(r%20=%200.9)_mpc.html)
[6](https://davidripsen.github.io/code_Smart_Charging/_figures/efficiency_id10885.html)
[7](https://davidripsen.github.io/code_Smart_Charging/_figures/mediods_vs_scenarios.html)
[8](https://davidripsen.github.io/code_Smart_Charging/_figures/Montas%20Smart%20Charge%20(dif.%20plug-in)%20of%20vehicle%20=%20853_mpc.html)
[9](https://davidripsen.github.io/code_Smart_Charging/_figures/Multi-Day%20Smart%20Charge%20(h%20=%204%20days)%20of%20vehicle%20=%20853_mpc.html)
[10](https://davidripsen.github.io/code_Smart_Charging/_figures/Perfect%20Foresight%20of%20vehicle%20=%20853_mpc.html)
[11](https://davidripsen.github.io/code_Smart_Charging/_figures/resultsBoxplot.html)
[12](https://davidripsen.github.io/code_Smart_Charging/_figures/resultsBoxplot2.html)
[13](https://davidripsen.github.io/code_Smart_Charging/_figures/spotprices.html)
[14](https://davidripsen.github.io/code_Smart_Charging/_figures/Stochastic%20Smart%20Charge%20(h%20=%204%20days)%20of%20vehicle%20=%20853_mpc.html)
[15](https://davidripsen.github.io/code_Smart_Charging/_figures/Stochastic-kMediods%20SC%20(h%20=%204%20days)%20of%20vehicle%20=%20853_mpc.html)
[16](https://davidripsen.github.io/code_Smart_Charging/_figures/use_curves_id13923.html)
[17](https://davidripsen.github.io/code_Smart_Charging/_figures/Covariance_matrix_of_residuals_Carnot.html)
[18](https://davidripsen.github.io/code_Smart_Charging/_figures/diurnal_weekly_CABLE_PLUGGED_IN_AT.html)
[19](https://davidripsen.github.io/code_Smart_Charging/_figures/diurnal_weekly_PLANNED_PICKUP_AT.html)
[20](https://davidripsen.github.io/code_Smart_Charging/_figures/diurnal_weekly_RELEASED_AT.html)
[21](https://davidripsen.github.io/code_Smart_Charging/_figures/Histograms_of_residuals_per_timestep_Carnot.gif)
[22](https://davidripsen.github.io/code_Smart_Charging/_figures/Mean_of_residuals_per_timestep_Carnot.html)
[23](https://davidripsen.github.io/code_Smart_Charging/_figures/PlainProfile_id10885.html)
[24](https://davidripsen.github.io/code_Smart_Charging/_figures/PlainProfile_id24727.html)
[25](https://davidripsen.github.io/code_Smart_Charging/_figures/PredictionMovie_Carnot.gif)
[26](https://davidripsen.github.io/code_Smart_Charging/_figures/Samples_from_multivariate_normal_distribution_Carnot.html)

## Abstract
A key aspect of transitioning to weather-driven energy sources is utilising the flexibility of energy consumption to meet the weather-driven time of production. An important example of such flexibility is the charging of electric vehicles (EVs). EV charging can often be postponed or advanced to times when the grid is in plenty, for example, overnight. This process is popularly called *Smart Charging*. This thesis provides models, methods and code for solving the Smart Charging problem of minimising costs.

The thesis shows that the Smart Charging problem can be solved by a simple linear program which is very fast to solve. For controlling the charging optimally over time, *Model Predictive Control* is suggested for solving the linear program with a *receding horizon* of up to six days of length. External data sources are used for price forecasts and historical EV behavioural data.

For comparison of various models, a novel measure is proposed: *Relative Total Cost*, which defines the relative difference from the total costs of having *Perfect Foresight* relative to *Dumb Charging* (as opposed to Smart Charging), charging immediately upon plug-in. This provides a framework for assessing all Smart Charging algorithms in a range from 0 % to 100 %.

Both deterministic and stochastic model predictive controllers are developed. The deterministic models treats the price forecast as the ground truth. This proves to fool the model into taking worse decisions than the Day-Ahead model, which only acts on the known prices 11-35 hours ahead.

To model the inherent uncertainty of the price forecasts, the Stochastic Multi-Day Ahead model is established. It is based on building a probabilistic forecast on top of the point forecast, from which statistical scenarios of the price residuals can be sampled. By defining the two-stage stochastic program of the problem, the model learns that forecasts are inherently uncertain and proves to improve performance relative to the deterministic Multi-Day Ahead model on the training set. A test set is obtained to estimate the out-of-sample performance, on which the stochastic model degrades in performance, calling for an adaptive estimation of the covariance structure in the residuals. 

Ultimately, the Day-Ahead Model, which only optimises on known prices, provides the best results on both training and test set. The cost of charging with the Day-Ahead model are much closer to that of the perfect charge plan (Perfect Foresight) than the worst possible charge plan (Dumb Charging). The average savings per vehicle is estimated to 1337 DKK (28 %) per year, but it requires a data input design which enables user flexibility

