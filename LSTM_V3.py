# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:50:42 2021

@author: Razaghi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xls = pd.ExcelFile('C:/Users/Razaghi/Desktop/Black-Litterman/Basenew.xlsx')
df = pd.read_excel(xls, 'Feuil1')
df=df.set_index('Date')



def normalize(data, name, windows):
    data=data[[name]].values
    ts_train_len=int(len(data)*0.8)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    ts_train = scaled_data[0:ts_train_len, :]
    y_test = data[ts_train_len:, :]
    x_train = []
    y_train = []
    for i in range(windows, len(ts_train)):
        x_train.append(ts_train[i-windows:i, 0]) 
        y_train.append(ts_train[i, 0]) 
        if i <= windows:
            print(x_train)
            print(y_train)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    ts_test = scaled_data[ts_train_len-windows:]
    x_test = []
    for i in range(windows, len(ts_test)):
        x_test.append(ts_test[i-windows:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return  y_train, y_test, x_train, x_test, scaler

y_train, y_test, x_train, x_test, scaler = normalize(df, name='SPXT Index', windows=5)

        
def lstm(y_train, x_train, x_test, scaler): 
    
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.optimizers import Adam
    minimum_hid_nodes = 655 
    # Achitecture du LSTM 
    LSTM_model = Sequential()
    LSTM_model.add(LSTM(units = minimum_hid_nodes, input_shape=(x_train.shape[1], 1),return_sequences=True,activation='tanh'))
    LSTM_model.add(LSTM(units = minimum_hid_nodes, return_sequences = True,activation='tanh'))    
    LSTM_model.add(LSTM(units = minimum_hid_nodes))
    LSTM_model.add(Dense(units = 1))
    LSTM_model.compile(loss='mean_squared_error', optimizer= Adam(learning_rate = 0.001))
    LSTM_model.fit(x_train, y_train, epochs=30, batch_size=100)
    # Prediction
    predict = LSTM_model.predict(x_test)
    predict = scaler.inverse_transform(predict)   
    return predict, LSTM_model

predict, LSTM_model = lstm(y_train, x_train, x_test, scaler)
    


def data_pred_plot(prediction):
    data_pred = pd.DataFrame(columns = ['observed', 'prediction'], index=np.arange(0,int(len(y_test)),1))
    data_pred['observed'] =  y_test
    data_pred['prediction'] = prediction
    from keras.metrics import RootMeanSquaredError
    m = RootMeanSquaredError()
    m.update_state(np.array(data_pred['observed']),np.array(data_pred['prediction']))    
    return (m.result().numpy(), data_pred.plot())    

data_pred_plot(predict)  


                             
def forecasting(t, model, name):   
       days = t
       forecast = np.array([])
       last = x_test[-1]
       for i in range(days):
           currentforecast = LSTM_model.predict(np.array([last]))
           print(currentforecast)
           last = np.concatenate([last[1:], currentforecast])
           forecast = np.concatenate([forecast, currentforecast[0]])
       forecast = scaler.inverse_transform([forecast])[0]
       import datetime
       from datetime import timedelta
       dicts = []
       currentdate = df[name].index[-1]
       for i in range(days):
           currentdate = currentdate + timedelta(days=1)
           dicts.append({'forecast':forecast[i], "Date": currentdate})
       forcast_data = pd.DataFrame(dicts).set_index("Date")
       return(forcast_data)

forecast = forecasting(t=30, model=LSTM_model, name='SPXT Index')
forecast



def data_forecast_plot(data, name):
    data=data[name]
    len_train=int(len(data)*0.8)    
    observed = pd.DataFrame(data.iloc[len_train:,])
    Date = observed.index
    prediction = pd.DataFrame(predict,columns=[name], index=Date)
    plt.figure(figsize=(8,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=8)
    plt.ylabel(name, fontsize=8)
    plt.plot(observed[name])
    plt.plot(prediction[name])
    plt.plot(forecast['forecast'])
    plt.legend(['Observed', 'Predict' ,'forecast'], loc='lower right')
    return plt.show()
    
data_forecast_plot(df, 'SPXT Index')
