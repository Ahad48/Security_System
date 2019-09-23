import cv2
import numpy as np


boundaries = [110, 50, 50], [130, 255, 255],#red


def colour(hsv, lower, upper,color):
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow(color, res)

#cap = cv2.VideoCapture(0)

frame = cv2.imread("/Users/ahad/PycharmProjects/python_project/venv/src/pokemon.jpg")

hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
while True:
    #red
    colour(hsv, np.array(boundaries[0]), np.array(boundaries[1]), 'red')

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()