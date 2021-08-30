import cv2
import os

name= input("Enter the Name of the person: ")
namedir= r"C:\Users\ronak\PycharmProject\FacePy\trainingImages" + "\\" + name
os.mkdir(namedir)
i=1
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame=video.read()
    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Capturing ',resized_img)
    #cv2.imshow("Capturing", cv2.flip(frame,1))
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    capture=input("Press C or c to capture Picture \nType exit to terminate - ")
    if capture == 'C' or capture=='c':
        img_item=str(i)+".jpg"
        os.chdir(namedir)
        cv2.imwrite(img_item,frame)
        i+=1 
    elif capture=='exit' or capture=='Exit':
        break

video.release()
cv2.destroyAllWindows()