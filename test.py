import cv2

l = cv2.imread("l.jpg")
r = cv2.imread("r.jpg")

cv2.imshow("", cv2.resize(r, None, fx=1/2, fy=1/2))
cv2.waitKey()