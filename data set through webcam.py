import cv2
import sys
cpt = 0

vidStream = cv2.VideoCapture(0)

while True:
    ret,frame = vidStream.read()
    cv2.imshow("Test Frame",frame)

    cv2.imwrite(r"D:\python project datasets\images\0\image%04i.jpg" %cpt,frame)
    cpt +=1

    if cv2.waitKey(10)==ord('q'):
        break
    
