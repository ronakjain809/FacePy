import cv2
import os
import numpy as np
from datetime import datetime
import faceRecognition as fr

# This module takes images  stored in disband performs face recognition
test_img = cv2.imread('TestImages/Ronak.jpg')  # test_img path
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)

# Comment belows lines when running this program second time.Since it saves training.yml file in directory
#faces, faceID = fr.labels_for_training_data('trainingImages')
#face_recognizer = fr.train_classifier(faces, faceID)
#face_recognizer.write('trainingData.yml')



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

name = {0: "Raj", 1: "Shivam", 2: "Ronak"}  # creating dictionary containing names for each label

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if (confidence > 44):  # If confidence more than 44 then don't print predicted face text on screen
        continue
    fr.put_text(test_img, predicted_name, x, y)

def markAttendance(predicted_name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if predicted_name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{predicted_name},{dtString}')
markAttendance(predicted_name)
resized_img = cv2.resize(test_img, (1000, 1000))
cv2.imshow("face detection tutorial", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
