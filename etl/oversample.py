import pickle
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


df = pd.read_pickle('../dataset/discretized_dataset')
df_label =  np.array(df.pop('taken')) 

train, test, train_labels, test_labels = train_test_split(df,
                                         df_label, 
                                         stratify = df_label,
                                         test_size = 0.1, 
                                         random_state = 42)

# Imputation of missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='minority', random_state=7)

# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = sm.fit_sample(train, train_labels)

divided_dataset = {'train': oversampled_trainX,
                   'test': test,
                   'train_labels': oversampled_trainY,
                   'test_labels': test_labels
                  }

with open('../dataset/oversampled_dataset', 'wb') as file:
    pickle.dump(divided_dataset, file)



