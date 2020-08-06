# Loading libraries:
import numpy
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

# Loading Dataset:

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshaping & Reloading Dataset:

img_rows, img_cols = 28, 28
num_classes = 10

def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_dataset()

#for i in range(3):
#    img = x_train[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()


# Defining model:

model=Sequential()
model.add(Dense(units=128, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train model:
model.fit(x_train, y_train, validation_data=(x_test,y_test), verbose=1, epochs=50, batch_size=128)

#Saving moodel:
model.save('ANN_Model.h5')
