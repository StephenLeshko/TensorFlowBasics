import os 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
import os.path
tfds.disable_progress_bar()

#metadata is data that describes the ACTUAL data
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 2 images from the dataset
for image, label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

IMG_SIZE = 160 #reduce images to 

#function that reduces size of images
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

#using formatter function on data
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

#viewing shapes of data
for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

for img, label in train.take(2):
    print("New shape:", img.shape)

#model utilization

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')

print('\nSummary\n')
base_model.summary()

for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape)

#makes it so you cannot alter the layers
base_model.trainable = False

#classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()

#training the model

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])
initial_epochs = 1
validation_steps = 20 #

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
    epochs=initial_epochs,
    validation_data=validation_batches)

acc = history.history['accuracy']
print('Accuracy\n\n', acc)

if os.path.isfile('dogs_vs_cats.h5') is False:
    model.save('dogs_vs_cats.h5') #saved weights, architecture, optimizer, and state of optimizer
#     #saves everything
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

#can also save the model as a JSON string
json_string = new_model.to_json() #needs to be retrained though
print(json_string) #needs to be compiled and optimized

#can also just save the weights

# model.save_weights('dogs_vs_cats_weights.h5')
# model2 = Sequential([
#     #layers
#     ])
# model2.load_weights('dogs_vs_cats_weights.h5')
# model2.get_weights()