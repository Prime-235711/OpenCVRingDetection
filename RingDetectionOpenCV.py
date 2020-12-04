import cv2
import numpy as np

# built-in cam is 0
# webcam is 1
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


def nothing():
    pass


cv2.namedWindow('Median Blur Slider Window')
cv2.createTrackbar('Median Blur Slider', 'Median Blur Slider Window', 21, 150, nothing)
cv2.namedWindow('U Threshold Window')
cv2.createTrackbar('U Threshold', 'U Threshold Window', 100, 254, nothing)


while True:
    ret, frame = cap.read()
    # Convert BGR to YUV
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)


    MedianBarPos = cv2.getTrackbarPos('Median Blur Slider', 'Median Blur Slider Window')
    UBarPos = cv2.getTrackbarPos('U Threshold', 'U Threshold Window')

    

    # blur the image
    yuvMedian = cv2.medianBlur(yuv, MedianBarPos)
    rawMedian = cv2.medianBlur(frame, MedianBarPos)

    # split the 3 channels
    y, u, v1 = cv2.split(yuv)

    ret, testMaskInv = cv2.threshold(u, UBarPos, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(testMaskInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) > 0):
        maxArea = 0
        maxAreaContour = -1
        for i in range(len(contours)):
            if(cv2.contourArea(contours[i]) > maxArea):
                maxArea = cv2.contourArea(contours[i])
                maxAreaContour = i

        area = cv2.contourArea(contours[maxAreaContour])
        x, y, width, height = cv2.boundingRect(contours[maxAreaContour])
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 5)
        #print(area)
        print(' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height))

    # display the masks

    cv2.imshow('Raw Input', frame)
    cv2.imshow('rawMedian', rawMedian)
    cv2.imshow('U', u)
    cv2.imshow('testMaskinv', testMaskInv)


    # press ESC to quit
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
