# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:54:20 2021

@author: rodion kovalenko
"""

# LSTM for crypto currencies prediction
import pandas as pd
import numpy as np
import pathlib
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import keras.optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import datetime
from pathlib import Path
import os
from numpy import array
import scipy.stats

def annotate_points(x_data, y_data, ax, interval_between = 10):
    for i, (x, y) in enumerate(zip(x_data, y_data)):   
        if (i % interval_between == 0):
            if not math.isnan(y):       
                label ="{:.2f}".format(y)               
            
                ax.annotate(label, # this is the text
                             (x,y), # these are the coordinates to position the label
                             textcoords="offset points", # how to position the text
                             xytext=(0, 2), # distance from text to points (x,y)
                             ha='left') # 

def predict(num_prediction, model, n_seq, n_steps, n_features, look_back, close_data):
    prediction_list = close_data[-1]
       
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]  
        x = x.reshape((-1, n_seq, n_steps, n_features))
        # x = x.reshape((-1, 1, look_back, 1))
        out = model.predict_on_batch(x)[0][0]
        # print('prediction {}'.format(out))
        prediction_list = np.append(prediction_list, out)
    # print('prediction list')
    # print(prediction_list)
    # print(prediction_list.shape)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list


def predict_dates(last_date, num_prediction):   
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

def save_plot(directory):
    datatime = datetime.datetime.today()
    datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year) + '-' + str(datatime.hour) + '-' + str(datatime.minute)
    filename = directory + '/' + currency + '-' + datestr + '.png';
        
    if os.path.exists(filename): os.remove(filename)
        
    Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(filename)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def inverse_transform(data):
    data = np.reshape(data, (-1, 1))
    #renormalize with minMaxScaler
    data = scaler.inverse_transform(data)
    #reshape in 1D
    data = np.reshape(data, (-1))
    return data

def my_distribution(min_val, max_val, mean, std):
    scale = max_val - min_val
    location = min_val
    # Mean and standard deviation of the unscaled beta distribution
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    # Computation of alpha and beta can be derived from mean and variance formulas
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    # Not all parameters may produce a valid distribution
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    # Make scaled beta distribution with computed parameters
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)


num_epochs = 10
interval_between_predicted = 10
#7 Tage = 48 * 7 = 336
num_prediction = 336
# we have univariate variable that is why n_features = 1
intervals = 10
n_features = 1
n_seq = 16
#time window
n_steps = 1024
look_back = n_steps
look_back_focast = n_steps
learning_rate =0.001


current_abs_path = str(pathlib.Path(__file__).parent.resolve())

datatime = datetime.datetime.today()
datestr = str(datatime.day) + '.' + str(datatime.month) + '.' + str(datatime.year)
save_plot_dir = current_abs_path + '/predictions/prediction-lstm-conv-norm/' + str(datestr)

print(current_abs_path)
data = pd.read_json(current_abs_path + '/cryptocurrency_rates_history.json')
currency_pairs = data[['full_date', 'pair', 'ask']]
currencies = currency_pairs['pair'].drop_duplicates()
scaler = MinMaxScaler()

for currency in currencies:
    if currency == 'ETH-EUR':
    # if currency is not None:
        n_seq = 16
        #time window
        n_steps = 1024
        currency_pairs_eth = currency_pairs['pair'] == currency
        data_filtered = currency_pairs[currency_pairs_eth]
        currency_data_all = data_filtered['ask'].values
        currency_dates = pd.to_datetime(data_filtered['full_date'])

        print('shape of data')
        print(data_filtered.shape)
        currency_dates[:] = np.reshape(currency_dates, (-1))
      
        #reshape in 2D
        close_data = data_filtered['ask'].values.reshape((-1, 1))
        close_data = np.reshape(close_data[close_data != 0], (-1, 1))
        close_data_plot = close_data[:].reshape((-1))
        #normalize data with minMaxScaler
        close_data = scaler.fit_transform(close_data)
        #reshape again in 1D
        close_data = close_data.reshape((-1))
        
        train_size = int(len(close_data) * 0.7)
        
        close_data_all, close_data_all_y = split_sequence(close_data, n_steps)    
        close_data_train, close_data_y = split_sequence(close_data[0: train_size], n_steps)
        test_data_x, test_data_y = split_sequence(close_data[train_size:], n_steps)
        
        n_steps = int(n_steps/n_seq)

        close_data_all = np.reshape(close_data_all, (close_data_all.shape[0], n_seq, n_steps, n_features)) 
        close_data_train = np.reshape(close_data_train, (close_data_train.shape[0], n_seq, n_steps, n_features))    
        test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], n_seq, n_steps, n_features))
        min_x = np.min(close_data)
        max_x = np.max(close_data)
        m = np.mean(close_data)
        sdt = np.std(close_data)
        
        saved_model_path = '/saved_models/lstm-conv-norm/' + currency
        saved_model_dir = current_abs_path + saved_model_path   
        print(saved_model_path)
        print(saved_model_dir)
    
        
        #build RNN
        if os.path.isdir(saved_model_dir):
            print('model was already saved')
            model = load_model(saved_model_dir)
        else:
            print('model was not saved')
            model = Sequential()
            model.add(TimeDistributed(Conv1D(filters=512, kernel_size=16, activation='relu'), input_shape=(None, n_steps, n_features)))
            model.add(TimeDistributed(MaxPooling1D(pool_size=32)))
            model.add(TimeDistributed(Flatten()))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            # model = Sequential()             
            # model.add(LSTM(100, input_shape=(n_steps, n_features)))
            # model.add(Dropout(0.2))
            # model.add(Dense(1))
           
        
        #train the model
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='mse')
        history = model.fit(close_data_all,
                            close_data_all_y,
                            epochs=num_epochs,
                            verbose=1,
                            shuffle=True,
                            batch_size = 32
                            )
        
        #save the trained model
        model.save(saved_model_dir) 
        
        train_predict = model.predict(close_data_all)
        test_predict = predict(len(close_data_all) - len(close_data_train) - 1, model, n_seq, n_steps, n_features, look_back_focast, close_data_train)
     
        close_data = np.reshape(close_data, (-1))
        pred_list = predict(num_prediction - 1, model, n_seq, n_steps, n_features, look_back_focast, close_data_all) 
        
    
        #generate random input for prediction
        # distribution = my_distribution(min_x, max_x, m, sdt)        
        # x_predict = distribution.rvs(size=num_prediction * n_steps)
        # x_predict = np.reshape(x_predict, (-1, n_steps, n_features))
        # x_predict = np.append(close_data_all, x_predict).reshape((-1, n_steps, n_features))
        # x_predict = model.predict(x_predict)
        # pred_list = x_predict[-num_prediction:]
        forecast_dates = np.array([datetime.datetime.today() + datetime.timedelta(minutes=30*x) for x in range(0, num_prediction)])
        
        #denormalize data
        train_predict = inverse_transform(train_predict)
        test_predict = inverse_transform(test_predict)       
        close_data = inverse_transform(close_data)        
        forecast = inverse_transform(pred_list)   
        
        prediction_dates = currency_dates    
            
        #plot the data     
        intervals = (np.max(close_data_plot) - np.min(close_data_plot)) / 10
        plt.figure(figsize=(30, 10))
        ax = plt.gca()
        
        annotate_points(prediction_dates, train_predict, ax, intervals)
        annotate_points(forecast_dates, forecast, ax, interval_between_predicted)    
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.set_yticks(np.arange(np.min(close_data_plot), np.max(close_data_plot), intervals))   
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.set_xlim([datetime.datetime.today() - datetime.timedelta(2), forecast_dates[-1] + datetime.timedelta(2)])
        ax.set_ylim(min(forecast), max(forecast))
        
        plt.plot(currency_dates, close_data)
        plt.plot(prediction_dates[-len(train_predict):], train_predict)
        # plt.plot(currency_dates[-len(test_predict):], test_predict)       
        plt.plot(forecast_dates, forecast, marker = 'o')
        
        additional_info = "Currency: " + currency + "\n" + '\n epoch: ' + str(num_epochs) + '\n window size: ' + str(n_steps)
            
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.figtext(.9, 0.9, additional_info)
        plt.legend(['original', 'trained', 'prediction'])
        ax.grid(True)
        save_plot(save_plot_dir)
        
        # plt.show()

















