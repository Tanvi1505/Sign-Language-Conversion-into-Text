import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I" , "K", "J", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z","OK","Thank you"]

# Define the duration of the timeslot in seconds
timeslot_duration = 60  # 60 seconds

# Record the start time of the timeslot
start_time = cv2.getTickCount() / cv2.getTickFrequency()

while True:
    # Check if the timeslot has elapsed
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    elapsed_time = current_time - start_time
    if elapsed_time >= timeslot_duration:
        print("Timeslot ended.")
        break  # Exit the loop if the timeslot duration has been reached

    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction,index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 0, 255), 2)
        cv2.rectangle(imgOutput,(x - offset,y-offset),(x+ w+offset, y+h+offset), (255,0,255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

