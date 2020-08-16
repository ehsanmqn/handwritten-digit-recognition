#Load libraries:
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

#Load model:
model = tf.keras.models.load_model('ANN_Model.h5')

#Load Dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Test model:
t = x_test[6].reshape(1,784,)
img = x_test[6].reshape((28,28))
p = model.predict_classes(t)
plt.imshow(img, cmap="Blues")
plt.show()
print('Correct digit:' ,y_test[6])
print('Predicted digit:' ,p)
