import pickle 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

RSEED = 50
LABEL = 'taken'

# Load in data
df = pd.read_pickle('../dataset/quantiles_discretized_dataset')

# Extract the labels
labels = np.array(df.pop(LABEL))
#train_y_2 = to_categorical(train_df_2.diabetes)


# 30% examples in test data
train_X, test_X, train_y, test_labels = train_test_split(df,
                                         labels, 
                                         stratify = labels,
                                         test_size = 0.1, 
                                         random_state = RSEED)

# Imputation of missing values
train_X = train_X.fillna(train_X.mean())
test_X = test_X.fillna(test_X.mean())

# Features for feature importances
features = list(train_X.columns)

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
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=200, callbacks=[early_stopping_monitor], verbose=1)

#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
#test_y_predictions = model.predict(test_X)

pickle.dump(model, open('dnn_model_keras', 'wb'))
dict_dataset = {"train_X":train_X, "test_X": test_X, "train_y":train_y, "test_labels":test_labels}
pickle.dump(dict_dataset, open('dict_dataset', 'wb'))


test_probs = model.predict(test_X)
test_predictions = np.argmax(test_probs, axis=1)
train_labels = train_y
train_probs=  model.predict(train_X)
train_predictions = np.argmax(train_probs, axis=1)
#print(confusion_matrix(test_labels, y_pred))


from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(test_labels, predictions, probs, train_labels, train_predictions, train_probs):
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(str(metric.capitalize())+ 'Baseline: ' + str(round(baseline[metric], 2)) + ' Test: '+ str(round(results[metric], 2)) + ' Train: ' +str(round(train_results[metric], 2)))
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 8))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves');
    plt.show();

    
evaluate_model(test_labels, test_predictions, test_probs, train_labels, train_predictions, train_probs)

plt.savefig('roc_auc_curve_50k_2.png')

