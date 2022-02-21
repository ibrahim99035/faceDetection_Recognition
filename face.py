import cv2 as cv
import numpy as np
import pickle as pk
#XML cascade file reading
face_cascade = cv.CascadeClassifier('haar.xml')
#recognizer :
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
#using pickels to save label IDs
labels = {}
with open('labels.pickle', 'rb') as f:
    loading_labels =  pk.load(f)
    labels = {v:k for k,v in loading_labels.items()}
#capture web cam
cap = cv.VideoCapture(0)
while (True) : 
    # capture frames frame by frame
    ret, frame = cap.read()

    #gray scale frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #detect the face 
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for(x, y, w, h)in face : 
        #printing coordinates
        #print(x,y,w,h)
        #picture 
        color_pic = frame[y:y+h, x:x+w]
        gray_pic  = gray[y:y+h, x:x+w]
        #-----------------------------------------------------------------------------------------
        #Recognition :
        #deep learn model to predict 
        id_, conf = recognizer.predict(gray_pic)
        if conf >= 45 and conf <= 85 :
            #print(labels[id_])
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            strok = 2
            cv.putText(frame, name, (x,y), font, 1, color, strok, cv.LINE_AA)

        #-----------------------------------------------------------------------------------------
        #capture the face in image from web cam
        
        img_item = "my-image.png"
        cv.imwrite(img_item, color_pic)

        #darwaing a rectange frame around faces
        color = (100, 210, 44) #BGR
        thic = 3
        cv.rectangle(frame, (x, y), (x + w, y + h), color, thic)

    # displaying process
    cv.imshow('Cam', frame)
    if cv.waitKey(20) & 0xFF == ord('q') :
        break

#release the capture 
cap.release()
cv.destroyAllWindows()
