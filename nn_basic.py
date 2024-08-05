# python program that creates a neural network to predict whether a person will have diabetes or not
# the neural network is trained on the Pima Indians Diabetes dataset
# the dataset includes females at least 21 years old of Pima Indian heritage
# the dataset includes 768 observations and 8 attributes
# 614 (80%) of the observations will be used for training the neural network
# 154 (20%) of the observations will be used for testing the neural network

# import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load in the training and testing datasets
dataset = pd.read_csv('diabetes.csv')
x = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# build the neural network using keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=100, batch_size=32)

# evaluate the model and print its accuracy
_, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy on testing data: {accuracy*100}')