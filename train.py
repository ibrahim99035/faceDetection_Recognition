from cProfile import label
from copyreg import pickle
import os
import numpy as np
import cv2 as cv
from PIL import Image
import pickle as pk

#XML cascade file reading
face_cascade = cv.CascadeClassifier('haar.xml')

#recognizer :
recognizer = cv.face.LBPHFaceRecognizer_create()

BASE_Dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_Dir, "TrainImages")

#for creating IDs :
current_id = 0
label_id = {}

x_train  = []
y_labels = []

for root, dirs, files in os.walk(image_dir) :
    for file in files :
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg") :
            path = os.path.join(root, file)
            Mylabel = os.path.basename(root).replace(" ", "-").lower()
            #print(Mylabel, path)
            pil_image = Image.open(path).convert("L") # gray scale
            #resize images for training
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") # converting it into numpy array
            #print(image_array)
            #------------------------------------------------------
            # for ID
            if not Mylabel in label_id :
                label_id[Mylabel] = current_id
                current_id += 1
            id_ = label_id[Mylabel]
            #print(label_id)
            #------------------------------------------------------
            #detect the face from image array
            face = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)
            for(x, y, w, h) in face :
                myImage = image_array[y : y + h, x : x + w]
                x_train.append(myImage)
                y_labels.append(id_)

#-------------------------------------------------------------------------------------------------------------
#print(y_labels)
#print(x_train)
#-------------------------------------------------------------------------------------------------------------
#using pickels to save label IDs
with open('labels.pickle', 'wb') as f:
    pk.dump(label_id, f)

#-------------------------------------------------------------------------------------------------------------
#recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")

#-------------------------------------------------------------------------------------------------------------
