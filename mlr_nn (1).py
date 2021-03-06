
#pip uninstall pandas_profiling


#pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

#Import the libraries such as pandas, NumPy.

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport    #imports pandas profiling library


#To read the dataset from the GoogleDrive
data = pd.read_csv('/content/gdrive/My Drive/Dataset3.csv')

#Print the data
data

#The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2)
#Renaming columns
data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

#To print a concise summary of a DataFrame, its memory usage and data types
data.info()

#pandas_profiling.ProfileReport(df)
#To see the data as Two-dimensional, size-mutable, potentially heterogeneous tabular data
df = pd.DataFrame(data)
df

#To get an overview of variables and its distribution
import pandas_profiling
pandas_profiling.ProfileReport(data)

#To know the correlation between variables in numbers
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)

#As the heatmap is difficult to read, next step is to format it and checking the correlation again
#Change number format in correlations
pd.set_option('display.float_format',lambda x: '{:,.2f}'.format(x) if abs(x) < 10000 else '{:,.0f}'.format(x))

data.corr()

#heating load and cooling are equally important outputs to be predicte, So we need to see the Correlation between inputs and outputs
plt.figure(figsize=(5,5))
sns.pairplot(data=data, y_vars=['cooling_load','heating_load'],
             x_vars=['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                     'orientation', 'glazing_area', 'glazing_area_distribution',])
plt.show()

#Preprocessing & Data Transformation
#Checking missing values
data.isnull().sum()

#Summary statistics/Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset???s distribution, excluding NaN values.
data.describe()

#Normalizing the inputs and set the output to obtain a better scale, because each feature has a different scale
from sklearn.preprocessing import Normalizer
nr = Normalizer(copy=False)

X = data.drop(['heating_load','cooling_load'], axis=1)
X = nr.fit_transform(X)
y = data[['heating_load','cooling_load']]

#Before training the models,praparing the input and output using Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

#creating Model Evaluation function to evaluate our model using R squared (R2 score)
def evaluate(model, test_features, test_labels):
    from sklearn.metrics import r2_score
    predictions = model.predict(test_features)
    R2 = np.mean(r2_score(test_labels, predictions))
    print('R2 score = %.3f' % R2)
    return r2_score

"""**As we saw from the histograms above, that the data in our dataset is discrete.Its a categorical data but in numbers.The Tree based algorithms need to be used to expect the best model using this type of data.**

**Creating 2 basic models and then optimze each models using Hyperparameter Search technique. The model we used are:
-Decission Tree Regression, Random Forest Regression

**1. Decission Tree Regressor**
"""

#Import decision tree regressor
from sklearn.tree import DecisionTreeRegressor
# Create decision tree model 
dt_model = DecisionTreeRegressor(random_state=123)
# Apply the model
dt_model.fit(X_train, y_train)
# Predicted value
y_pred1 = dt_model.predict(X_test)

#R2 score before optimization
R2_before_dt= evaluate(dt_model, X_test, y_test)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#Visualize the heating load output before optimization
plt.figure(figsize = (5,5))
ax1.plot(range(0,len(X_test)),y_test.iloc[:,0],'o',color='red',label = 'Actual Values')
ax1.plot(range(0,len(X_test)),y_pred1[:,0],'X',color='yellow',label = 'Predicted Values')
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Heating Load')
ax1.set_title('Heating  Load Before Optimization')
ax1.legend(loc = 'upper right')

#Visualize the cooling load output before optimization
plt.figure(figsize = (5,5))
ax2.plot(range(0,len(X_test)),y_test.iloc[:,1],'o',color='green',label = 'Actual Values')
ax2.plot(range(0,len(X_test)),y_pred1[:,1],'X',color='blue',label = 'Predicted Values')
ax2.set_xlabel('Test Cases')
ax2.set_ylabel('Cooling Load')
ax2.set_title('Cooling Load Before Optimization')
ax2.legend(loc = 'upper right')

plt.show()

# Finding the best decision tree optimization parameters

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# Max Depth
dt_acc = []
dt_depth = range(1,11)
for i in dt_depth:
    dt = DecisionTreeRegressor(random_state=123, max_depth=i)
    dt.fit(X_train, y_train)
    dt_acc.append(dt.score(X_test, y_test))
ax1.plot(dt_depth,dt_acc)
ax1.set_title('Max Depth')

#Min Samples Split guarantees a minimum number of samples in a leaf
dt_acc = []
dt_samples_split = range(10,21)
for i in dt_samples_split:
    dt = DecisionTreeRegressor(random_state=123, min_samples_split=i)
    dt.fit(X_train, y_train)
    dt_acc.append(dt.score(X_test, y_test))
ax2.plot(dt_samples_split,dt_acc)
ax2.set_title('Min Samples Split')

plt.show()

#Minimum Sample Leaf specifies the minimum number of samples required to be at a leaf node.
plt.figure(figsize = (5,5))
dt_acc = []
dt_samples_leaf = range(1,10)
for i in dt_samples_leaf:
    dt = DecisionTreeRegressor(random_state=123, min_samples_leaf=i)
    dt.fit(X_train, y_train)
    dt_acc.append(dt.score(X_test, y_test))

plt.plot(dt_samples_leaf,dt_acc)
plt.title('Min Sample Leaf')

plt.show()

# Decision tree optimization parameters
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth' : [7,8,9],
              'min_samples_split': [16,17,18],
              'min_samples_leaf' : [6,7,8]}


#Create new model using the GridSearch
dt_random = GridSearchCV(dt_model, parameters, cv=10)

#Apply the model
dt_random.fit(X_train, y_train)

#View the best parameters
dt_random.best_params_

# Predicted value
y_pred1_ = dt_random.best_estimator_.predict(X_test)

#R2 score after optimization
dt_best_random = dt_random.best_estimator_
R2_after_dt= evaluate(dt_best_random, X_test, y_test)

#Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#Visualize the heating load output after optimization
plt.figure(figsize = (5,5))
ax1.plot(range(0,len(X_test)),y_test.iloc[:,0],'o',color='red',label = 'Actual Values')
ax1.plot(range(0,len(X_test)),y_pred1_[:,0],'X',color='yellow',label = 'Predicted Values')
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Heating Load')
ax1.set_title('Heating  Load After Optimization')
ax1.legend(loc = 'upper right')

#Visualize the cooling load output after optimization
plt.figure(figsize = (5,5))
ax2.plot(range(0,len(X_test)),y_test.iloc[:,1],'o',color='green',label = 'Actual Values')
ax2.plot(range(0,len(X_test)),y_pred1_[:,1],'X',color='blue',label = 'Predicted Values')
ax2.set_xlabel('Test Cases')
ax2.set_ylabel('Cooling Load')
ax2.set_title('Cooling Load After Optimization')
ax2.legend(loc = 'upper right')

plt.show()

"""**2. Random Forest Regressor**
A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees.The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees.
Random Forest has multiple decision trees as base learning models. It randomly perform row sampling and feature sampling from the dataset forming sample datasets for every model. This part is called Bootstrap.
"""

#Import random forest regressor
from sklearn.ensemble import RandomForestRegressor
# Create random forest model 
rf_model = RandomForestRegressor(random_state=123)
# Apply the model
rf_model.fit(X_train, y_train)
# Predicted value
y_pred2 = rf_model.predict(X_test)

#R2 score before optimization
R2_before_rf= evaluate(rf_model, X_test, y_test)

#Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#Visualize the heating load output before optimization
plt.figure(figsize = (5,5))
ax1.plot(range(0,len(X_test)),y_test.iloc[:,0],'o',color='red',label = 'Actual Values')
ax1.plot(range(0,len(X_test)),y_pred2[:,0],'X',color='yellow',label = 'Predicted Values')
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Heating Load')
ax1.set_title('Heating  Load Before Optimization')
ax1.legend(loc = 'upper right')

#Visualize the cooling load output before optimization
plt.figure(figsize = (5,5))
ax2.plot(range(0,len(X_test)),y_test.iloc[:,1],'o',color='green',label = 'Actual Values')
ax2.plot(range(0,len(X_test)),y_pred2[:,1],'X',color='blue',label = 'Predicted Values')
ax2.set_xlabel('Test Cases')
ax2.set_ylabel('Cooling Load')
ax2.set_title('Cooling Load Before Optimization')
ax2.legend(loc = 'upper right')

plt.show()

# Finding the best random forest optimization parameters

f, axarr = plt.subplots(2, 2)

# Max Depth
rf_acc = []
rf_depth = range(1,11)
for i in rf_depth:
    rf = RandomForestRegressor(random_state=123, max_depth=i)
    rf.fit(X_train, y_train)
    rf_acc.append(rf.score(X_test, y_test))
axarr[0, 0].plot(rf_depth,rf_acc)
axarr[0, 0].set_title('Max Depth')

#Min Samples Split
rf_acc = []
rf_samples_split = range(10,21)
for i in rf_samples_split:
    rf = RandomForestRegressor(random_state=123, min_samples_split=i)
    rf.fit(X_train, y_train)
    rf_acc.append(rf.score(X_test, y_test))
axarr[0, 1].plot(rf_samples_split,rf_acc)
axarr[0, 1].set_title('Min Samples Split')

#Min Sample Leaf
rf_acc = []
rf_samples_leaf = range(1,10)
for i in rf_samples_leaf:
    rf = RandomForestRegressor(random_state=123, min_samples_leaf=i)
    rf.fit(X_train, y_train)
    rf_acc.append(rf.score(X_test, y_test))

axarr[1, 0].plot(rf_samples_leaf,rf_acc)
axarr[1, 0].set_title('Min Sample Leaf')

#N Estimator
rf_acc = []
rf_estimators = range(50,59)
for i in rf_estimators:
    rf = RandomForestRegressor(random_state=123, n_estimators=i)
    rf.fit(X_train, y_train)
    rf_acc.append(rf.score(X_test, y_test))

axarr[1, 1].plot(rf_estimators,rf_acc)
axarr[1, 1].set_title('N Estimator')

plt.show()

# Random forest optimization parameters(Hyperparameter tuning)
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth' : [6,7,8],
              'min_samples_split': [11,12,13],
              'min_samples_leaf' : [4,5,6],
              'n_estimators': [49,50,51]}


#Create new model using the GridSearch
rf_random = GridSearchCV(rf_model, parameters, cv=10)

#Apply the model
rf_random.fit(X_train, y_train)

#View the best parameters
rf_random.best_params_

# Predicted values
y_pred2_ = rf_random.best_estimator_.predict(X_test)

#R2 score after optimization
best_random_rf = rf_random.best_estimator_
R2_after_rf= evaluate(best_random_rf, X_test, y_test)

#Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#Visualize the heating load output after optimization
plt.figure(figsize = (5,5))
ax1.plot(range(0,len(X_test)),y_test.iloc[:,0],'o',color='red',label = 'Actual Values')
ax1.plot(range(0,len(X_test)),y_pred2_[:,0],'X',color='yellow',label = 'Predicted Values')
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Heating Load')
ax1.set_title('Heating  Load After Optimization')
ax1.legend(loc = 'upper right')

#Visualize the cooling load output after optimization
plt.figure(figsize = (5,5))
ax2.plot(range(0,len(X_test)),y_test.iloc[:,1],'o',color='green',label = 'Actual Values')
ax2.plot(range(0,len(X_test)),y_pred2_[:,1],'X',color='blue',label = 'Predicted Values')
ax2.set_xlabel('Test Cases')
ax2.set_ylabel('Cooling Load')
ax2.set_title('Cooling Load After Optimization')
ax2.legend(loc = 'upper right')

plt.show()

"""**As R2 score is not always the best indicator of fit. R squared (R2 score) will always increase as we add more independent variables ??? but adjusted R2 will decrease if we add an independent variable that does not help the model**"""

# create a fitted model with all features
import statsmodels.formula.api as smf
data2=data.copy()
lm1 = smf.ols(formula='heating_load ~ relative_compactness + surface_area + wall_area + roof_area + overall_height + orientation + glazing_area + glazing_area_distribution', data=data2).fit()

# Summarizing the fitted model
lm1.summary()

# creating a fitted model with all features excluding the feature "orientation"
lm2 = smf.ols(formula='heating_load ~ relative_compactness + surface_area + wall_area + roof_area + overall_height + glazing_area + glazing_area_distribution', data=data2).fit()

# Summarizing the fitted model
lm2.summary()



"""# **Technique-2 DENSE NEURAL NETWORK on EE01-Dataset(Dataset-1)**"""

# Importing the libraries such as pandas, NumPy, tensorflow etc.
import pandas as pd
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


data = pd.read_csv('/content/gdrive/My Drive/Dataset3.csv')
df = pd.DataFrame.from_dict(data);
df

# The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2)
# Rename columns 
data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',
                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
data

# We have implemented utility function for actual and predicted values plot
# We also have implemented data conversion functions such as normalizing data
def format_output(data):
    y1 = data.pop('heating_load')
    y1 = np.array(y1)
    y2 = data.pop('cooling_load')
    y2 = np.array(y2)
    return y1, y2

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

def plot_difference(y_true, y_pred, title='Neural Networks stats'):
    plt.scatter(y_true, y_pred)
    plt.title(title)

    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')
    plt.axis('scaled')

    plt.plot([-70, 70], [-70, 70])
    plt.show()

def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(model_history.history[metric_name], color='red', label=metric_name)
    plt.plot(model_history.history['val_' + metric_name], color='blue', label='val_' + metric_name)
    plt.show()

df = df.sample(frac=1).reset_index(drop=True)

# We will split our dataset into train and test with 70% train and 30% test
train, test = train_test_split(df, test_size=0.3)
train_stats = train.describe()

# After dataset split, we are getting heating_load and cooling_load as the two outputs and format train and test data as np arrays
train_stats.pop('heating_load')
train_stats.pop('cooling_load')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

# Normalize the training and testing data
norm_train_X = norm(train)
norm_test_X = norm(test)

# Define model layers where we have layer with 128 dimensional output space and rectified linear unit activation function
input_layer = Input(shape=(len(train .columns),))
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)

# From second dense, we will use Heading load output
hl_output = Dense(units='1', name='hl_output')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

# From third dense, we will use Cooling load output
cl_output = Dense(units='1', name='cl_output')(third_dense)

# We have defined model using input layer and heating_load , cooling_load output as a list 
nn_model = Model(inputs=input_layer, outputs=[hl_output, cl_output])

print(nn_model.summary())

# We are using stochastic gradient descent(SGD) optimizer with a learning rate of 0.001
# Then compiling the model with loss functions for both heating load and cooling load outputs
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
nn_model.compile(optimizer=optimizer,
              loss={'hl_output': 'mean_squared_error', 'cl_output': 'mean_squared_error'},
              metrics={'hl_output': tf.keras.metrics.RootMeanSquaredError(),
                       'cl_output': tf.keras.metrics.RootMeanSquaredError()})

# Model is trained on batch size of 15 and 600 epochs
model_history = nn_model.fit(norm_train_X, train_Y,
                    epochs=600, batch_size=15, validation_data=(norm_test_X, test_Y))

# In the model test step, we are calculating loss and mean squared error for heating and cooling load outputs
loss, hl_loss, cl_loss, hl_rmse, cl_rmse = nn_model.evaluate(x=norm_test_X, y=test_Y)
print("Loss = {}, hl_loss = {}, hl_mse = {}, cl_loss = {}, cl_mse = {}".format(loss, hl_loss, hl_rmse, cl_loss, cl_rmse))

# We have plotted loss and mean squared error graphs
Y_pred = nn_model.predict(norm_test_X)
plot_difference(test_Y[0], Y_pred[0], title='Heating Load')
plot_difference(test_Y[1], Y_pred[1], title='Cooling Load')
plot_metrics(metric_name='hl_output_root_mean_squared_error', title='HL RMSE', ylim=10)
plot_metrics(metric_name='cl_output_root_mean_squared_error', title='CL RMSE', ylim=10)

