from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# labels correspond to these classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plot one training sample
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalize input
train_images = train_images / 255.0
test_images = test_images / 255.0

# plot some training sample and label
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # flatten 2D(28x28) to 1D
    keras.layers.Dense(128, activation='relu'),     # FC layer, 128 units
    keras.layers.Dense(10, activation='softmax')    # output layer, 10 classification with softmax
])

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # why this function?
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=10)

# evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
