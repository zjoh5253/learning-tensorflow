import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# get the fashion_mnist dataset
data = keras.datasets.fashion_mnist

print("loading data")
(train_images, train_labels), (test_images, test_labels) = data.load_data()
print("finished loading data")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()

