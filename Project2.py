import requests
import numpy as np
import cv2
nPlateCascade = cv2.CascadeClassifier("Resources\haarcascade_russian_plate_number.xml")
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
cam.set(10,150)
count =0
while True:
    img_res = requests.get("http://192.168.43.83:8080/shot.jpg")
    img_arr = np.array(bytearray(img_res.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr,-1)
    img = cv2.resize(img,(640,480))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in nplate:
        area = w*h
        if area>500:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),3)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("Roi",imgRoi)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Resources/Scanned/NoPlate_"+str(count)+".jpg",imgRoi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_ITALIC,2,(0,0,255),2)
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        count +=1
