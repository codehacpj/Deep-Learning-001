# RNN 

# Part 1: Data preprocessing 

# libraries import 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# importing data
data = pd.read_csv("Google_Stock_Price_Train.csv")
data_train = data.iloc[:,1:2].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
sc_train=sc.fit_transform(data_train)

# finding the number of time Steps
X_train = []
y_train = []
for i in range(60,1257):
    X_train.append(sc_train[i-60:i,0])
    y_train.append(sc_train[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)


#Reshaping to prepare X_train to be fit into LSTM nets keras documentation gives
#information about what shape to give to it.(reshape it to (batch_size-->no of entries,timesteps-->number of columns,input_dim--->number of indecators))

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Part 2: Building RNN

#import models and libraries
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

#initialize the RNN
regressor = Sequential()

#Adding layer1(LSTM)
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding layer2(LSTM)
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#Adding layer4(LSTM)
regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

#Adding layer5(output layer Dense)
regressor.add(Dense(units=1))

#fitting the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')





















