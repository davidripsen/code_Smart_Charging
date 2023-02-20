# Smart Charging Electric Vehicles
Welcome!
In this repository you will find all the code and models for the Masters Thesis of David Ribberholt Ipsen.
Checkout [visualisations](https://davidripsen.github.io/code_Smart_Charging/) to see some of the results.

## Code
The main code is located in ´MPC/FunctionCollection.py´

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
A key aspect of transitioning to weather-driven energy sources is utilising the flexibility of energy consumption to meet the weather-driven time of production. An important example of such flexibility is the charging of electric vehicles (EVs). EV charging can often be postponed or advanced overnight when the grid is in plenty. This process is popularly called \textit{Smart Charging}. This thesis provides models, methods and code for solving the Smart Charging problem of minimising costs.

It turns out that the Smart Charging problem can be modelled by a linear program which is very fast to solve. For controlling the charging optimally over time, \textit{Model Predictive Control} is suggested for solving the linear program with a \textit{receding horizon} of up to 6 days of length. When considering the next six days in determining how much to charge \textit{here-and-now}, one will need to predict the expected usage, plug-in hours and price. For the latter, open-source forecast is utilised. For the former, EV behavioural data has been provided by \href{www.monta.com}{Monta}.

%Throughout the thesis, both deterministic and stochastic models are developed to model the Smart Charging problem.
For comparison of various models, a novel measure is proposed: \textit{Relative Total Cost}, which defines the relative difference from the total costs of having Perfect Foresight relative to \textit{Dumb Charging}, charging immediately upon plug-in. All other models are assessed in this range from 0 \% to 100 \%.

Both deterministic and stochastic models are developed. The deterministic models assume the price forecasts to be true. This proves to fool the models into taking worse decisions, thereby increasing cost with the increasing horizon. To model the uncertainty of the price forecasts, stochastic models are introduced. They are based on building a probabilistic forecast head on top of the external point forecast, from which statistical scenarios of the price residuals can be sampled. By defining the two-stage stochastic program of the problem, the model learns that forecasts are inherently uncertain and proves to improve performance relative to the deterministic models.

Ultimately, the thesis finds that the Day-Ahead Model, which only optimises on the known prices, surprisingly performs best. In other words, it has not been possible to improve performance by including price forecasts in the decision-making. This model is, however, still more sophisticated than industry state-of-the-art Smart Charging algorithms. Further complexities are suggested for future work, but better forecasts of price and usage are deemed necessary to improve the models.