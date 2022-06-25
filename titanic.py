from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output


import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(y_train)
# print(dftrain.describe())
# print(y_train.describe())

#showp plot of certain column variable
# dftrain.age.value_counts().plot(kind = 'barh')
# plt.show()

#Feature columns

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = [] #what is feed to the model
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # .unique() gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

dftrain['sex'].unique()

def make_input_fn(data_df, label_df, num_epochs=1, shuffle=True, batch_size=32): #label_df is the y_train thing
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#making the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#training
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
print('Saving the model\n\n')

serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec([feature_columns]))

print('input done')
linear_est.export_saved_model('modelfromEstimator', serving_input_fn)


print('\ndone')

#checking
# clear_output()