import cv2
import time

face_detector= cv2.CascadeClassifier('faceIndex.xml')

face_id =  input("Informe o id da indexação")


img = cv2.imread("images/image3.jpg",0)
faces = face_detector.detectMultiScale(img, 1.3, 5)

if(len(faces)!=0):
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imwrite("dataset/Subject." + str(face_id) + ".0" +".jpg", img[y:y+h,x:x+w])
    cv2.imshow('frame',img)

cv2.destroyAllWindows()