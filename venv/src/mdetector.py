import cv2, time, pandas
from datetime import datetime
face_casc=cv2.CascadeClassifier("C:/Users/Nilanjana/PycharmProjects/test/haarcascade_frontalface_default.xml")

first_frame=None

video=cv2.VideoCapture(0)
face_status_list=[0 , 0]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

while True:
    
    check, frame= video.read()
    face_status = 0
    gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray1,(21,21),0)

    faces=face_casc.detectMultiScale(gray1,
    scaleFactor=1.5,minNeighbors=5)

    for x, y, w, h in faces:
        gray1=cv2.rectangle(gray1,(x,y),(x+w,y+h),(255,255,0),3)
        face_status = 1

    
#To capture the first frame
    if first_frame is None:
        first_frame=gray
        continue

#Calculate the difference between the first static frame and the rest
    delta_frame=cv2.absdiff(first_frame,gray)
    
#Create threshold frame where difference is white
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    #assigns color white(255) when it is above a certain threshold(30)
    #thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
    

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#If contour is less than 1000 then it is irrelevant thus, continue
    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue

        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),3)
    face_status_list.append(face_status)

    if face_status_list[-1]==1 and face_status_list[-2]==0:
        times.append(datetime.now())
    if face_status_list[-1]==0 and face_status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Face Detection",gray1)
    #cv2.imshow("l",delta_frame)
    #cv2.imshow("la",thresh_frame)
    #cv2.imshow("Motion Detection",frame)
    key=cv2.waitKey(1)
   

    if key==ord('q'):
        if face_status==1:
            times.append(datetime.now())
        break
print(face_status_list)
print(len(times))
for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")
video.release()


cv2.destroyAllWindows
