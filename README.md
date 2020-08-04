# DE4S_Model
DE4S is a timeseries forecasting model capable of forecasting time-varying seasonal structures. The model is still under development, however it has been shown to succesfully forecast with competitive accuracy when compared to other widely used forecasting methods.

To run a forecast, required parameters are a dataframe containing the exogenous variable and a date column (df), and exogenous variable name (exog), a date header (date), the initial level (level), level smoothing (alpha), trend (trend), and trend smoothing (beta) parameters. <br><br>

Initialize:<br>
model = SeasonalSwitchingModel(df, exog, date, level, alpha, trend, beta).

Fit:<br>
fitted_model = model.fit_seasonal_switching_model()

Predict:<br>
fitted_model.predict(n_steps)

Plot seasonal structures:<br>
fitted_model.plot_seasonal_structures()<br><br>


Package dependencies are found in the requirements.txt.

A paper describing the method is included in the repository.
