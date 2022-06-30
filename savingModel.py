#test used to save and load models using
#the keras API

import os 
import tensorflow as tf
from tensorflow import keras
import PIL.Image
print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#Testing to view images
# for i in range(2):
#     print('\n\n', train_images[i])
#     img = PIL.Image.fromarray(train_images[i])
#     img.show()
    

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#create a model
def create_model():
    model = tf.keras.Sequential([
        #each layer is defined by # of inputs, activation function, shape usually
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        
    return model


    
    

model = create_model()
model.summary()

#create a training checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

# Train the model with the new callback
model.fit(train_images, 
    train_labels,  
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback])  # Pass callback to training

os.listdir(checkpoint_dir)

#laoding weights to new model
model = create_model()
#evaluate tests it on images
loss, acc = model.evaluate(test_images, test_labels, verbose=2)

#loading weights
model.load_weights(checkpoint_path)

#training a model and saving unique checkpoints at 
#every 5 epochs

#can manually save the weights too
model.save_weights('place where stored')

#saved model format
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('loc')

#retrieve savedModel
new_model = tf.keras.models.load_model('loc')
#can also save it as an .h5 file using save after fitting
