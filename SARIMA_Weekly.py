##SARIMA Weekly
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error

actual_array = list()
predicted_array = list()

#Reading the data
dataset = pd.read_csv('/content/gdrive/My Drive/dataset2.txt', sep=';', low_memory=False,header=0, usecols = ["Date","Time","Global_active_power"], 
                       infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
#Data Preprocessing
dataset.replace('?', np.nan, inplace=True)
dataset = dataset.astype('float32')
weekly=dataset.resample('W').mean()
train = weekly[0:159]
test = weekly[159:]

#Training the model
model = SARIMAX(train, order=(1, 1, 1),seasonal_order=(1, 1, 0, 48),enforce_stationarity=False,enforce_invertibility=False)
#Testing the mode
results = model.fit()
prediction = results.predict(start='2010-01-03', end='2010-11-28')
prediction.columns = ['Date_Time','Global_active_power']

for i in range(0, len(test)):
  actual_array.append(test["Global_active_power"][i])
  predicted_array.append(prediction[i])

err_magnitude = mean_squared_error(actual_array, predicted_array, squared=False)
print("Root Mean Square Error for SARIMA: ",err_magnitude)

#Plotting
f, diag = plt.subplots(3, figsize=(15, 10), sharex=True)
diag[0].plot(actual_array, color='black', label='actual')
diag[0].plot(predicted_array, color='red', label='prediction')
diag[1].plot(predicted_array, color='red', label='prediction')
diag[1].set_title('Predicted')
diag[2].plot(actual_array, color='black', label='Actual')
diag[2].set_title('Actual')
diag[0].set_title('Actual vs Predicted Energy')
diag[0].set_ylabel('Global_active_power')
diag[0].legend()
plt.show()