import cv2
import tensorflow as tf
import numpy as np


def DetectedBox(row, col, numRows, numCols, boxSizeW, boxSizeH, cardSizeW, cardSizeH, imageSizeW, imageSizeH):
    w = boxSizeW * imageSizeW / cardSizeW
    h = boxSizeH * imageSizeH / cardSizeH
    x = (imageSizeW - w) / ((numCols - 1)) * (col)
    y = (imageSizeH - h) / ((numRows - 1)) * (row)
    return int(x), int(y), int(w), int(h)

image = cv2.imread("card.jpg")
image = cv2.resize(image, (480, 302), interpolation=cv2.INTER_AREA)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="findfour.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data[0] = image
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

boxes = []
for row in range(0, 34):
    for col in range(0, 51):
        if output_data[0][row][col][1] > 0.5:
            confidence = output_data[0][row][col][1]
            x, y, w, h = DetectedBox(row, col, 34, 51, 80, 36, 480, 302, 480, 302)
            boxes.append([confidence, x, y, w, h])

boxes.sort(reverse=True)
boxes = boxes[0:20]

for item in boxes:
    print(">> ", item)

item = boxes[0]
x = item[1]
y = item[2]
w = item[3]
h = item[4]

image = cv2.rectangle(image, (y, y+h), (x, x+w), (255, 0, 0), 2)
cv2.imshow("", image)
cv2.waitKey()