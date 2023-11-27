import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
#%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

import plotly.graph_objects as go
import plotly.express as px
tf.random.set_seed(1234)
csv_path ='E:\Reinforcing-a-Monitoring-System-of-a-Regulated-River-using-ML-main\Datasets\Charleston_Gh.csv'
water_data = pd.read_csv(csv_path)
water_data
water_data['timestamp'] = pd.to_datetime(water_data['Date'])
water_data['water_discharge'] = water_data['WD']
#water_data['data_qualification_code'] = water_data['value']
water_data
# drop unnecessary colums
water_data =water_data.drop(['Date','WD'], axis=1)
water_data
fig = px.line(water_data.reset_index(), x = 'timestamp', y = 'water_discharge')
fig.update_xaxes(rangeslider_visible = True)
fig.show()
# Set the DATE column as the index.
new_water_data = water_data.set_index('timestamp')
new_water_data.head()
new_water_data.index
new_water_data
# Check for missing values
print('Total data point:')
print(len(new_water_data))
print('Total num of missing values:') 
print(new_water_data.water_discharge.isna().sum())
print('')
# finding duplicate
# new_water_data[new_water_data.index.duplicated()]
new_water_data.index.duplicated()
# Drop rows with duplicate index values
final_water_data = new_water_data.loc[~new_water_data.index.duplicated(), :]
final_water_data
# Missing time series row
dt = pd.Series(data = pd.date_range(start=final_water_data.index.min(), end=final_water_data.index.max(), freq = 'H'))
idx = pd. DatetimeIndex(dt)
final_water_data2 = final_water_data.reindex(idx)
final_water_data2
df = final_water_data2
df.head()
#Verify that there are no empty cells in any column. The output should be all zeroes.
import numpy as np
df.isin([np.nan, np.inf, -np.inf]).sum()
# Check for missing values
print('Total data point:')
print(len(df))
print('Total num of missing values:') 
print(df.water_discharge.isna().sum())
print('')
# Locate the missing value
new_water_data_missing_date = df.loc[df.water_discharge.isna() == True]
print(new_water_data_missing_date)
# locating the index of the missing values
i = df[df.water_discharge.isna() == True].index
print(i)
# replacing missing values in water_discharge column
# with median of that column
df['water_discharge'] = df['water_discharge'].fillna(df['water_discharge'].mean())
#print(df[i], df.water_discharge[i])
# Check for missing values
print('Total data point:')
print(len(df))
print('Total num of missing values:') 
print(df.water_discharge.isna().sum())
print('')
# Plot out the time series
df.plot(figsize=(12,8))
final_water_data2 =df
data_count = len(final_water_data2)
print(data_count)
train_size = int(0.8*data_count)
print(train_size)
test_size = data_count-train_size
print(test_size)
training_set = df.iloc[:train_size, 0:1].values
training_set
len(training_set)
testing_set = df.iloc[train_size:, 0:1].values
testing_set.shape
df['water_discharge'][:train_size].plot(figsize=(16,4),legend=True)
df['water_discharge'][train_size:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set ','Test set'])
plt.title('Water Discharge')
plt.show()
# Feature scaling to optimize the training set
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
# Create a data structure with 1 timesteps and 1 output
# Format it into a three-dimensional array for use in our LSTM model
X_train = []
y_train = []
for i in range(1, train_size):
    X_train.append(training_set_scaled[i-1:i, 0]) # Appending previous 1 data.
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (LSMT Model needs to be 3- dimensional) following format (values, time-steps, 1 dimensional output).
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
import keras
def fit_model(model):
    
#-----Early stop implementation-----  
#     early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
#                                                patience = 10)
#     history = model.fit(X_train, y_train, epochs = 50,  
#                         validation_split = 0.2,
#                         batch_size = 16, shuffle = False, 
#                         callbacks = [early_stop])
    
# ------ without Early stop ----
    history = model.fit(X_train, y_train, epochs = 50,  
                        validation_split = 0.2,
                        batch_size = 16, shuffle = False
                        )    
    return history
# Create GRU model
def create_gru(units):
    model = Sequential()
    # Input layer
    model.add(GRU (units = units, return_sequences = True, 
    input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.02)) 
    # Hidden layer
    model.add(GRU(units = units)) 
    model.add(Dropout(0.02))
    model.add(Dense(units = 1)) 
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model
model_gru = create_gru(64)
history_gru = fit_model(model_gru)
# Create BiLSTM model
def create_bilstm(units):
    model = Sequential()
    # Input layer
    model.add(Bidirectional(
              LSTM(units = units, return_sequences=True), 
              input_shape=(X_train.shape[1], X_train.shape[2])))
    # Hidden layer
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    #Compile model
    model.compile(optimizer='adam',loss='mse')
    return model
model_bilstm = create_bilstm(64)
history_bilstm = fit_model(model_bilstm)
# creating LSTM model
model_lstm = Sequential()

# Adding first LSTM layer and some dropout Dropout regularisation
model_lstm.add(LSTM(units=100, return_sequences=True, input_shape = (X_train.shape[1], 1)))
model_lstm.add(Dropout(0.1))

# Adding second LSTM layer and some dropout Dropout regularisation
model_lstm.add(LSTM(units=100, return_sequences=True))
model_lstm.add(Dropout(0.1))

# Adding third LSTM layer and some dropout Dropout regularisation
model_lstm.add(LSTM(units=100, return_sequences=True))
model_lstm.add(Dropout(0.1))

# Adding fourth LSTM layer and some dropout Dropout regularisation
model_lstm.add(LSTM(units=100, return_sequences=True))
model_lstm.add(Dropout(0.1))

# Adding fifth LSTM layer and some dropout Dropout regularisation
model_lstm.add(LSTM(units=100))
model_lstm.add(Dropout(0.1))

# Adding the Output Layer
model_lstm.add(Dense(units=1))

# Compiling the model
# Because we're doing regression hence mean_squared_error
model_lstm.compile(optimizer = 'adam', loss = 'mean_squared_error')
history_lstm = fit_model(model_lstm)
# Evaluating The Model
def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
 
plot_loss (history_gru, 'GRU')
plot_loss (history_bilstm, 'BiLSTM')
plot_loss (history_lstm, 'LSTM')
# We will concatenate the dataset and then scale them, as the last 1 data needs to be from training set
dataset_total = df['water_discharge']
inputs = dataset_total[len(dataset_total) - len(testing_set) - 1:].values
inputs = inputs.reshape(-1,1) # reshape(-1,1) means one column with all rows
inputs = sc.transform(inputs)

X_test = []
for i in range(1, test_size+1): # 1+5276(test size)
    X_test.append(inputs[i-1:i, 0])

X_test = np.array(X_test)
# 3D format for LSTM
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
inputs.shape
X_test.shape
#preict the model
# Make prediction
def prediction(model):
    prediction = model.predict(X_test)
    prediction = sc.inverse_transform(prediction)
    return prediction
prediction_gru = prediction(model_gru)
prediction_bilstm = prediction(model_bilstm)
prediction_lstm = prediction(model_lstm)
# Plot test data vs prediction
# Prediction is on top of real data
def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(12, 6))
    range_future = len(prediction)

    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Real data', color='b')
    plt.plot(np.arange(range_future), 
             np.array(prediction),label='Prediction',color='r')
    
    plt.title('Real data vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('water discharge')
 
plot_future(prediction_gru, 'GRU', testing_set)
plot_future(prediction_bilstm, 'BiLSTM', testing_set)
plot_future(prediction_lstm, 'LSTM', testing_set)
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions-actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape= np.mean(np.abs(errors/actual)*100)
    
    print(model_name + ':')
    print('Mean Absolute Error (MAE): {:.4f}'.format(mae))
    print('Mean Squared Error (MSE): {:.4f}'.format(mse))
    print('Root Mean Square Error(RMSE): {:.4f}'.format(rmse))
    print('Mean Absolute Percentage Error(MAPE): {:.4f}'.format(mape))
    print('')
evaluate_prediction(prediction_gru, testing_set, 'GRU')
evaluate_prediction(prediction_bilstm, testing_set, 'Bidirectiona LSTM')
evaluate_prediction(prediction_lstm, testing_set, 'LSTM')

from sklearn.metrics import r2_score

GRU_r2 = r2_score(testing_set, prediction_gru)
print(f'R2 score of GRU: {GRU_r2}')

BiLSTM_r2 = r2_score(testing_set, prediction_bilstm)
print(f'R2 score of BiLSTM: {BiLSTM_r2}')

LSTM_r2 = r2_score(testing_set, prediction_lstm)
print(f'R2 score of LSTM: {LSTM_r2}')