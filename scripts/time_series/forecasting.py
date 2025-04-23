# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:26:01 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from RegscorePy import aic
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error

###--------------Import and Format Data
file_path = r"C:\Users\Arjoh\Downloads\Assignment 6 Demo Data.csv"
df = pd.read_csv(file_path)
df.drop(columns=['Unnamed: 0', 'ID'], inplace=True) #Drop unnecessary columns
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M') #Convert Datetime column to a proper format
df = df.iloc[:11856] #Subset data to 2014 start

#Split into train and test
train = df.iloc[:10392].copy()
test = df.iloc[10392:].copy()

#Set index to timestamp for resampling
df.set_index('Datetime', inplace=True)
train.set_index('Datetime', inplace=True)
test.set_index('Datetime', inplace=True)

###--------------Aggregate by day (using sum instead of mean)
    #'D', 'W', 'M', 'Q', 'A' based on how time is divided
train = train.resample('D').sum()
test = test.resample('D').sum()

###--------------Plot data
plt.figure(figsize=(15, 8))
plt.plot(train.index, train['Count'], label="Train")
plt.plot(test.index, test['Count'], label="Test")
plt.title('Daily Ridership')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

###--------------Testing the Accuracy of Forecasts
# Assuming y_observed and y_predicted are defined
# Example placeholders (Replace with actual predicted values)
y_observed = test['Count']  # Actual values
y_predicted = np.random.uniform(y_observed.min(), y_observed.max(), size=len(y_observed))  # Dummy predicted values

#Explained Variance Score
evs = explained_variance_score(y_observed, y_predicted)

#Mean Absolute Error
mae = mean_absolute_error(y_observed, y_predicted)

#Mean Squared Error
mse = mean_squared_error(y_observed, y_predicted)

#Root Mean Squared Error
rmse = np.sqrt(mse)

#Akaike Information Criterion (AIC)
p = 1  # Adjust this value based on your model complexity
aic_value = aic.aic(y_observed.tolist(), y_predicted.tolist(), p)

###--------------Print results
print(f"Explained Variance Score (EVS): {evs:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Akaike Information Criterion (AIC): {aic_value:.4f}")
print('-' * 50)

###--------------Method 1 – A Naive Mode1
    #assumes that the next expected point is equal to the last observed point
#Get the last observed value in training data
last_observed_value = train['Count'].iloc[-1]

#Apply the last observed value to all test data points
test['naive_forecast'] = np.full(len(test), last_observed_value)

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['Count'], label='Train', color='blue')
plt.plot(test.index, test['Count'], label='Test', color='green')
plt.plot(test.index, test['naive_forecast'], label='Naïve Forecast', linestyle='dotted', color='red')
plt.legend(loc='best')
plt.title("Naïve Forecasting Method")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Validate Accuracy using RMSE
rmse = np.sqrt(mean_squared_error(test['Count'], test['naive_forecast'])).round(2)
print(f"Root Mean Squared Error (RMSE) for Naïve Forecast: {rmse}")

###--------------Method 2 – Simple Average
#forecast the next day based on the average of past observations
simple_avg_forecast = train['Count'].mean()

#Apply the mean to all test data points
test['avg_forecast'] = np.full(len(test), simple_avg_forecast)

#Plot the results
plt.figure(figsize=(12, 8))
plt.plot(train['Count'], label='Train', color='blue')
plt.plot(test['Count'], label='Test', color='green')
plt.plot(test.index, test['avg_forecast'], label='Simple Average Forecast', linestyle='dotted', color='red')
plt.legend(loc='best')
plt.title("Simple Average Forecasting")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Validate Accuracy using RMSE
rmse = np.sqrt(mean_squared_error(test['Count'], test['avg_forecast'])).round(2)
print(f"Root Mean Squared Error (RMSE) for Simple Average: {rmse}")

###--------------Method 3 – Moving Average
#Rolling average. Set p in rolling() for different SMAs
train_len = train.shape[0] # counts the nr of rows in df train
sma_data = pd.concat([train, test]) # the rolling average is across all data
y_hat_sma = sma_data.copy()
y_hat_sma['sma_forecast'] = sma_data['Count'].rolling(10).mean()

#Plot data
plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
plt.legend(loc='best')
plt.show()

#Validate Accuracy using RMSE
rms = np.sqrt(mean_squared_error(test['Count'],
y_hat_sma['sma_forecast'][train_len:])).round(2) #notice rounding
print(f"Root Mean Squared Error (RMSE) for Moving Average: {rms}")

#Set rolling window size
window_size = 10

#Apply SMA on training data
train['sma_forecast'] = train['Count'].rolling(window=window_size).mean()

#Forecast for the test period (use the last known SMA value from train)
sma_forecast_test = train['sma_forecast'].iloc[-1]
test['sma_forecast'] = np.full(len(test), sma_forecast_test)

#Plot forecasted data
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train', color='blue')
plt.plot(test['Count'], label='Test', color='green')
plt.plot(train['sma_forecast'], label=f'Training SMA ({window_size} days)', linestyle='dashed', color='red')
plt.plot(test.index, test['sma_forecast'], label='SMA Forecast (Future)', linestyle='dotted', color='black')
plt.legend(loc='best')
plt.title("Simple Moving Average (SMA) Forecasting")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Validate Accuracy using RMSE
rmse = np.sqrt(mean_squared_error(test['Count'], test['sma_forecast'])).round(2)
print(f"Root Mean Squared Error (RMSE) for SMA: {rmse}")
print('-' * 50)

###--------------Method 4 – Weighted moving average
#Define weights for the moving average (most recent value has the highest weight)
weights = np.array([0.15, 0.2, 0.25, 0.4])  #Older -> Newer

#Ensure weights equal to 1
weights = weights / weights.sum()

#Apply the weighted moving average on 'Count' column
df['WMA'] = df['Count'].rolling(window=4).apply(lambda x: np.dot(weights, x), raw=True)

#Display the first few rows
print(df[['Count', 'WMA']].head(10))

###--------------Method 5 – Single Exponential Smoothing
    #Single/Simple Exponential Smoothing (SES) is a time series forecasting method 
    #for univariate data without a trend or seasonality
#Prepare SES data
ses_data = pd.concat([train, test])['Count'].copy()

#Normalization
train = train.resample('D').sum()
test = test.resample('D').sum()

#Fit SES model
fit = SimpleExpSmoothing(ses_data).fit(smoothing_level=0.6, optimized=False)

#Generate predictions
ses_data_forecast = fit.fittedvalues  # In-sample predictions

#Future forecast for test period
future_forecast = fit.forecast(len(test))

#Plot data
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train', color='blue')
plt.plot(test['Count'], label='Test', color='green')
plt.plot(ses_data_forecast, label='SES Fitted (α=0.6)', color='red')
plt.plot(future_forecast, label='SES Forecast (Future)', linestyle="dashed", color='black')
plt.legend(loc='best')
plt.title("Single Exponential Smoothing (SES) Forecasting")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Validate Accuracy using RMSE
print('-' * 50)
rmse = np.sqrt(mean_squared_error(test['Count'], future_forecast)).round(2)
print(f"Root Mean Squared Error (RMSE) for SES: {rmse}")

###--------------Method 6 – Double Exponential Smoothing
#Prepare Holt-Winters data
holt_data = pd.concat([train, test])['Count']

#Normalization
train = train.resample('D').sum()
test = test.resample('D').sum()

##Fit Various Holt's Models
#Additive Model
fit1 = Holt(holt_data, initialization_method="estimated").fit(smoothing_level=0.8,
                                                               smoothing_trend=0.2,
                                                               optimized=False)
fcast1 = fit1.forecast(5).rename("Additive Model")

#Exponetial Model
fit2 = Holt(holt_data, exponential=True, initialization_method="estimated").fit(
    smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
fcast2 = fit2.forecast(5).rename("Exponential Trend")

#Damped Version of the Additive Model with ϕ optimized
fit3 = Holt(holt_data, damped_trend=True, initialization_method="estimated").fit(
    smoothing_level=0.8, smoothing_trend=0.2)
fcast3 = fit3.forecast(5).rename("Additive Damped Trend")

#Exponential Model with All Parameters Optimized
fit4 = Holt(holt_data, initialization_method="estimated").fit(optimized=True)
fcast4 = fit4.forecast(5).rename("Automatic")

#Plot models 1-3 on one chart
plt.figure(figsize=(12, 8))
plt.plot(holt_data, marker="o", color="black", label="Original Data")
plt.plot(fit1.fittedvalues, color="blue", label="Additive Model")
plt.plot(fcast1, marker="o", color="blue", linestyle="dashed")
plt.plot(fit2.fittedvalues, color="red", label="Exponential Trend")
plt.plot(fcast2, marker="o", color="red", linestyle="dashed")
plt.plot(fit3.fittedvalues, color="yellow", label="Damped Additive Trend")
plt.plot(fcast3, marker="o", color="yellow", linestyle="dashed")
plt.legend()
plt.title("Holt’s Forecasting Methods")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

#Validate Accuracy using RMSE
rmse = np.sqrt(mean_squared_error(holt_data, fit1.fittedvalues))
print(f"Root Mean Squared Error (RMSE) for Holt's Linear Trend: {rmse:.4f}")
print('-' * 50)

###--------------Method 7 – Holt-Winters Method
#Combine train and test data for Holt-Winters method
hw_data = pd.concat([train, test])

#Ensure data is correctly indexed
hw_data = hw_data['Count']

#Fit different Holt-Winters models
fit1 = SimpleExpSmoothing(hw_data, initialization_method="estimated").fit()
fit2 = Holt(hw_data, initialization_method="estimated").fit()
fit3 = Holt(hw_data, exponential=True, initialization_method="estimated").fit()
fit4 = Holt(hw_data, damped_trend=True, initialization_method="estimated").fit()
fit5 = Holt(hw_data, exponential=True, damped_trend=True, initialization_method="estimated").fit()

#Forecast 10 units into the future
fcast5 = fit5.forecast(10).rename("Fit5")

#Prepare results for output
params = [
    "smoothing_level",
    "smoothing_trend",
    "damping_trend",
    "initial_level",
    "initial_trend",]

#Create results DataFrame
results = pd.DataFrame(
    index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$l_0$", "$b_0$", "SSE"],
    columns=["SES", "Holt's", "Exponential", "Additive", "Multiplicative"],)

results["SES"] = [fit1.params.get(p, np.nan) for p in params] + [fit1.sse]
results["Holt's"] = [fit2.params.get(p, np.nan) for p in params] + [fit2.sse]
results["Exponential"] = [fit3.params.get(p, np.nan) for p in params] + [fit3.sse]
results["Additive"] = [fit4.params.get(p, np.nan) for p in params] + [fit4.sse]
results["Multiplicative"] = [fit5.params.get(p, np.nan) for p in params] + [fit5.sse]

#Display results table
print(results)

#Plot seasonally adjusted data
plt.figure(figsize=(12, 6))
plt.plot(hw_data, label="Original Data", color="black")

for fit, label in zip(
    [fit1, fit2, fit3, fit4, fit5],
    ["SES", "Holt's", "Exponential", "Additive", "Multiplicative"],):
    plt.plot(fit.fittedvalues, label=label)

plt.legend()
plt.title("Holt-Winters Model Fits")
plt.show()

