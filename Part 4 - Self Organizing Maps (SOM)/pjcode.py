#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(10,10,input_len=15,sigma=1.0,learning_rate=0.5,decay_function=None,random_seed=42)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#Visualising the map
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()

markers=['o','s']
colors=['r','g']

for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding frauds
mappings=som.win_map(X)
frauds = np.concatenate((mappings[(5,7)],mappings[(4,6)]),axis=0)
frauds = sc.inverse_transform(frauds)
print("The number of potential frauds is",format(len(frauds)))


#Now we will use the the data of potential frauds to build our dependent variable for ANN(supervised learning)

# Independent variable from dataset, excluding customer id(no use for fraud detection)
customers = dataset.iloc[:,1:].values

# To create the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1


# building ANN to find the probability of a customer to be a fraud
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)

y_pred = y_pred[y_pred[:,1].argsort()]
print("y_pred has the sorted values of probability for customers to be frauds.")
print(str(y_pred[-1,0]) +" has the highest probability of being fraud")


