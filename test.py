import cv2
import numpy as np

Vector = [[1,0],[1,1]]
v = [[1,2]]

# print(Vector[0])

# print(v[0][0]+Vector[0][0])

im = cv2.imread("./input/1T20.png",1)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(contours)