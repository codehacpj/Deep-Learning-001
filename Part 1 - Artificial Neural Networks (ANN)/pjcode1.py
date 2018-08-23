# Artificial Neural Networks

#Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing Tensorflow
#from website see notebook

# installing keras 
#pip install --upgrade keras

#Part1 - Data Preprocessing:
#Classification :

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #credit score,geography,gender,age,tenure,balance, no.of products, has credit card, activity,est.salary
y = dataset.iloc[:, 13].values

# Encoding categorical data (for gender and country)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#to escape the ordinality of the coded variable for the country avoiding this one for gender as when we remove one to escape dummy variable trap there won't be more than one variable for the gender feature..
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Escaping dummy variable trap, we remove one dummy variable.
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 ANN

#Importing the Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()

#adding the input layer and 1st hidden layer
classifier.add(Dense(output_dim=6,init = 'uniform',activation='relu', input_dim=11))

#adding new hidden layer.
classifier.add(Dense(output_dim=6,init = 'uniform',activation='relu'))

#Adding output layer (sigmoid function)
classifier.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to the Training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score = classifier.evaluate(X_test, y_test, batch_size=10)
print("The model has ", score[1]*100,"% accuracy!\n",sep="")


