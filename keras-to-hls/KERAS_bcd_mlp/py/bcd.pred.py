from __future__ import print_function

import keras
from keras.models import model_from_json

#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers.core import Dense
#from keras.layers.core import Activation
#from keras.optimizers import RMSprop
#
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Some helpers
def run_onehot_encoder(data_df):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    data = label_encoder.fit_transform(data_df)
    data = data.reshape(len(data), 1)
    data = onehot_encoder.fit_transform(data)
    return data

def print_array_to_h(data):
    f = open("inputs.h","w")

    f.write("input_t  data_str[N_INPUTS] = {")

    i=0
    for d in data:
        if i==0:
            f.write("%.12f" % d)
        else:
            f.write(", %.12f" % d)
        i=i+1
    f.write("};\n")
    f.close()

#
# File names
target = 'bcd'
model_file_json = '../../example-keras-model-files/KERAS_' + target + '_mlp.json'
model_file_h5 = '../../example-keras-model-files/KERAS_' + target + '_mlp_weights.h5'

# Model reconstruction from JSON file
with open(model_file_json, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(model_file_h5)

# Read CSV data
data_df = pd.read_csv('./dataset/wdbc.data')

# Preview the loaded data
#print(data.head())

# Splitting the data into features and labels
X_data_df = data_df.drop('diagnosis', axis=1).drop('id', axis=1)
y_data_df = data_df.diagnosis

# Splitting the data into train (70%) and test data (30%)
(X_train_df, X_test_df, y_train_df, y_test_df) = train_test_split(X_data_df, y_data_df, test_size=0.3, random_state=10)

# Preview the train and test data
#show_preview(X_train_df, X_test_df, y_train_df, y_test_df)

# Get dimensions
input_feature_size = X_test_df.shape[1] # number of input features
test_size = X_test_df.shape[0] # number of data points

# Reshaping the test data
X_test = X_test_df.values

# One-hot encoding of the labels
y_test = run_onehot_encoder(y_test_df)

# Dataset pre-processing
X_test = StandardScaler().fit_transform(X_test)

# Choose a sample
sample_index = 1
sample = X_test[sample_index]
label = y_test[sample_index]

# Run prediction
pred = model.predict(np.array([sample.reshape(30,)]))

# Some information
print('INFO: input shape: ', sample.shape)
print('INFO: one-hot encoding: ', label) # 2
print('INFO: predictions: ', pred[0])
print('INFO: top prediction: ', pred.argmax())

print_array_to_h(sample)


