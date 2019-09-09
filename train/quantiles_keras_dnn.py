import pickle 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout


from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import custom_plotting as cp

RSEED = 42
LABEL = 'taken'

with open('../dataset/oversampled_dataset', 'rb') as file:
    oversampled_dataset = pickle.load(file)
 
# 30% examples in test data
train_X = oversampled_dataset['train']
test_X = oversampled_dataset['test']
train_y = oversampled_dataset['train_labels']
test_labels = oversampled_dataset['test_labels']

#new sequential model
model = Sequential()

#number of features in training set
n_cols = train_X.shape[1]

#relu as activation function for hidden layers, whereas sigmoid was the choice for the last layer
model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))

#select adam optimizer and mse as loss function
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)

#class_weight = {1: 0.3,0: 0.7}

#train model
model.fit(train_X, 
          train_y, 
          validation_split=0.2, 
          epochs=100, 
          callbacks=[early_stopping_monitor], 
          verbose=1, 
          #class_weight=class_weight
         )



##############################################################
##################### Evaluate results #######################
##############################################################

test_probs = model.predict(test_X)
print(str(test_probs.min()))
print(str(test_probs.max()))
test_predictions = np.argmax(test_probs, axis=1)
train_labels = train_y
train_probs =  model.predict(train_X)
train_predictions = np.argmax(train_probs, axis=1)

    
cp.evaluate_model(test_labels, test_predictions, test_probs, train_labels, train_predictions, train_probs)

