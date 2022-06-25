import tensorflow as tf
from tensorflow import keras as keras
import tensorflowjs as tfjs

input_column = tf.feature_column.numeric_column("x")
a = tf.estimator.LinearClassifier(input_column, model_dir='model')


serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
  tf.feature_column.make_parse_example_spec([input_column]))

print('checkpoint:\n\n\n')
# Save Estimator as a tf model

def trainingDay():
    print('its training day')

a.train(trainingDay)

a.export_saved_model("modelFromEstimator", serving_input_fn)

# Import model as keras model
model = keras.models.load_model("modelFromEstimator")

# Save as tfjs model
j = tfjs.converters.save_keras_model(model, "tfjsmodel")
