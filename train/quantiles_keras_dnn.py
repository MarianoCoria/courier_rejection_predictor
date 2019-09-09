import pickle 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import custom_plotting as cp

RSEED = 42
LABEL = 'taken'

# Load in data
df = pd.read_pickle('../dataset/discretized_dataset')

# Extract the labels
labels = np.array(df.pop(LABEL))

# train and test datasets division
train_X, test_X, train_y, test_labels = train_test_split(df,
                                         labels, 
                                         stratify = labels,
                                         test_size = 0.2, 
                                         random_state = RSEED)

# Filling missing values
train_X = train_X.fillna(train_X.mean())
test_X = test_X.fillna(test_X.mean())

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)

class_weight = {1: 0.3,
                0: 0.7}

#train model
model.fit(train_X, 
          train_y, 
          validation_split=0.1, 
          epochs=100, 
          callbacks=[early_stopping_monitor], 
          verbose=1, 
          class_weight=class_weight
         )



##############################################################
##################### Evaluate results #######################
##############################################################

test_probs = model.predict(test_X)
print(str(test_probs.min()))
print(str(test_probs.max()))
test_predictions = np.argmax(test_probs, axis=1)
train_labels = train_y
train_probs=  model.predict(train_X)
train_predictions = np.argmax(train_probs, axis=1)

    
cp.evaluate_model(test_labels, test_predictions, test_probs, train_labels, train_predictions, train_probs)

