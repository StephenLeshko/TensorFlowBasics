import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
import tensorflow as tf
import numpy as np
import matplotlib as mpl
#testing how I can display images on my machine
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
name = url.split('/')[-1]
image_path = tf.keras.utils.get_file(name, origin=url)
img = PIL.Image.open(image_path)

print(img)
img.show()