import os
import cv2
import numpy as np
import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def main():
    model=load_model("model.h5")
    face_haar_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)
    while True:
        r,test_image=cap.read()
        if not r:
            continue
        grey_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        face_detected=face_haar_cascade.detectMultiScale(grey_image,1.32,5)

        for (x,y,w,h) in face_detected:
            cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=grey_image[y:y+w,x:x+h]
            roi_gray=cv2.resize(roi_gray,(224,224))
            img_px=image.img_to_array(roi_gray)
            img_px=np.expand_dims(img_px,axis=0)
            img_px=img_px/255

            predictions=model.predict(img_px)
            max_index=np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_image,predicted_emotion,(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        
        resized_img=cv2.resize(test_image,(1000,700))
        cv2.imshow('Facial emotion:', resized_img)

        if cv2.waitKey(10)==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows
if __name__=="__main__":
    main()