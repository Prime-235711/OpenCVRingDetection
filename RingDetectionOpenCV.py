import cv2
import numpy as np

# built-in cam is 0
# webcam is 1
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# A dummy method that the sliders require
def nothing(nothing):
    pass

# Creates the sliders with their windows
cv2.namedWindow('Median Blur Slider Window')
cv2.createTrackbar('Median Blur Slider', 'Median Blur Slider Window', 21, 150, nothing)
cv2.namedWindow('U Threshold Window')
cv2.createTrackbar('U Threshold', 'U Threshold Window', 100, 254, nothing)

# Main loop
while True:
    # Input currrent frame of webcam as frame
    ret, frame = cap.read()
    
    # Convert BGR color space to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Read values of the sliders
    MedianBarPos = cv2.getTrackbarPos('Median Blur Slider', 'Median Blur Slider Window')
    UBarPos = cv2.getTrackbarPos('U Threshold', 'U Threshold Window')

    # The input to the blur function must be odd.
    if(MedianBarPos % 2 == 0):
        MedianBarPos += 1

    # Blur the image using the median blur algorithm
    yuvMedian = cv2.medianBlur(yuv, MedianBarPos)
    rawMedian = cv2.medianBlur(frame, MedianBarPos)

    # Split the 3 channels
    y, u, v1 = cv2.split(yuvMedian)

    # A mask that removes all things not orange enough
    ret, RingMaskInv = cv2.threshold(u, UBarPos, 255, cv2.THRESH_BINARY_INV)

    # Gets all contours. Bassically all of the groups or orange
    contours, hierarchy = cv2.findContours(RingMaskInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finds the biggest contour. Hopefeully the ring.
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

    # Display the frames
    cv2.imshow('Raw Input', frame)
    cv2.imshow('rawMedian', rawMedian)
    cv2.imshow('U', u)
    cv2.imshow('testMaskinv', RingMaskInv)


    # Press ESC to quit
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
