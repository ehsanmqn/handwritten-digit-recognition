import cv2
import copy
import joblib
import numpy as np
from skimage.feature import hog
import tensorflow as tf


# Load CNN classifier
model = tf.keras.models.load_model('CNN_Model.h5')

# Load SVN classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image 
im = cv2.imread("card1.jpg")
cv2.imshow("", im)
cv2.waitKey()
# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using CNN and Linear SVM
output_image = copy.copy(im)
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(output_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1) 
    
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2 - 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2 - 2)
    roi = im_th[pt1:pt1+leng + 2, pt2:pt2+leng + 2]
    
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Recognize digit using CNN
    t = roi.reshape(1,28,28,1)
    p = model.predict_classes(t)
    cv2.putText(output_image, str(int(p)), (rect[0] - 10, rect[1] - 5),cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 0), 2)


    # Calculate the HOG features
    # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    # cv2.putText(output_image, str(int(nbr)), (rect[0] + 10, rect[1] - 5),cv2.FONT_HERSHEY_DUPLEX, .8, (0, 255, 255), 2)


# cv2.putText(output_image, "Yellow: SVM", (150, im.shape[0] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
cv2.putText(output_image, "Blue: CNN", (10, im.shape[0] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

cv2.imshow("Resulting Image with Rectangular ROIs", output_image)
cv2.waitKey()