#Load libraries:
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

#Load Dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Load CNN Model:
model = tf.keras.models.load_model('CNN_Model.h5')

#Test Model:
t = x_test[150].reshape(1,28,28,1)
img = x_test[150].reshape((28,28))
p = model.predict_classes(t)
plt.imshow(img, cmap="Blues")
plt.axis('off')
plt.show()
print('Correct digit:' ,y_test[150])
print('Predicted digit:' ,p)
