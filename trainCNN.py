# Loading libraries:
import numpy
import matplotlib.pyplot as plt
from keras.datasets import mnist
import keras 
from keras.layers import Dense, Conv2D,Dropout,MaxPooling2D,Flatten
from keras.models import Sequential 

img_rows, img_cols = 28, 28
num_classes = 10

# Load & Reshaping Dataset:
def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000,28,28,1)
    x_test = x_test.reshape(10000,28,28,1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_dataset()

# Define Model:
def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),  activation='relu',  input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

model = define_model()

# Train Model:
model.fit(x_train, y_train, validation_data=(x_test, y_test),verbose=1, epochs=25, batch_size=128)

#Evaluate Model:
score = model.evaluate(x_test, y_test, verbose=0)
print('Loss :', score[0] , '%')
print('Accuracy :', score[1]*100, '%')

#Save Model:
model.save('CNN_Model.h5')
