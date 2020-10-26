import numpy as numpy
import cv2
import os
import face_recognition as fr
#print(fr)

test_img=cv2.imread(r'C:\Users\Harsha\AppData\Local\Programs\Python\Python38\python project\face recognition\test_img2.jpeg')

faces_detected,gray_img=fr.faceDetection(test_img)
#cv2.imshow(faces_detected)
print("face Detected: ",faces_detected)

#training will being from here

faces,faceID=fr.labels_for_training_data(r'D:\python project datasets\images')
face_recognizer=fr.train_Classifier(faces,faceID)
face_recognizer.save(r'C:\Users\Harsha\AppData\Local\Programs\Python\Python38\python project\face recognition\trainingData.yml')

name={0:'Harsha'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)
resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("Face Detection",resized_img)
cv2.waitKey(0)
cv2.destoryAllWindows()


