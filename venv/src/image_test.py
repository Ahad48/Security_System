import cv2

first_frame = cv2.imread("/Users/ahad/PycharmProjects/python_project/venv/src/static.jpg")
frame = cv2.imread("/Users/ahad/PycharmProjects/python_project/venv/src/object.jpg")
first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
first_frame=cv2.GaussianBlur(first_frame,(21,21),0)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

delta_frame=cv2.absdiff(first_frame,gray)
thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
(cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for contour in cnts:
    if cv2.contourArea(contour) < 10000:
        continue
    status=1

    (x, y, w, h)=cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("Gray Frame",gray)
cv2.imshow("Delta Frame",delta_frame)
cv2.imshow("Threshold Frame",thresh_frame)
cv2.imshow("Color Frame",frame)


cv2.waitKey(0)
cv2.destroyAllWindows()