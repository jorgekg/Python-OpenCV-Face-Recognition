import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('train/train.yml')

cascadePath = "faceIndex.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    im=cv2.flip(im,1,0)

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id)
        if(Id == 1):
            Id = "Rick"
        elif(Id == 2):
            Id = "Daryl"
        elif(Id == 3):
            Id = "Jorge"
        elif(Id == 4):
            Id = "Vermelhao"
        elif(Id == 5):
            Id = "Miguel"
        
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

    cv2.imshow('im',im) 

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()