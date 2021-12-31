#ARIMA Monthly
from pandas import read_csv
import keras.models as models
import keras.layers as layers
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import matplotlib.pyplot as plt
%matplotlib inline

actual_array = list()
predicted_array = list()

#Reading the data
data = pd.read_csv('/content/gdrive/My Drive/dataset2.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
data.replace('?', np.nan, inplace=True)

#Data Preprocessing
data = data.astype('float32')
datas = data["Global_active_power"]
datas.replace(np.nan, 0, inplace=True)

#Fetching the monthly data
monthly=datas.resample('M').mean()
train = monthly[:36]
test = monthly[36:]

#Training the model
Arima_model=pm.auto_arima(train)

#Testing the model
predicted = pd.DataFrame(Arima_model.predict(n_periods=12))

#Calculating the RMSE
for i in range(0, len(test)):
  actual_array.append(test[i])
for i in range(0, len(predicted)):
  predicted_array.append(predicted[0][i])
err_magnitude = mean_squared_error(test, predicted, squared=False)
print("Root Mean Square Error for ARIMA: ",err_magnitude)
f, axes = plt.subplots(3, figsize=(15, 10), sharex=True)

#Plotting
axes[0].plot(actual_array, color='black', label='actual')
axes[0].plot(predicted_array, color='blue', label='prediction')
axes[0].set_title('Actual vs Predicted Energy')
axes[0].set_ylabel('Global_active_power')
axes[0].legend()

axes[1].plot(predicted_array, color='blue', label='prediction')
axes[1].set_title('Predicted')

axes[2].plot(actual_array, color='black', label='Actual')
axes[1].set_title('Actual')

plt.show()
