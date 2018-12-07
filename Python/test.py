#coding:utf-8
import cv2
import math

def main():
    img = cv2.imread("./input/example.png")
    for i in range(len(img)):
        for j in range(len(img[i])):
            for c in range(len(img[i,j])):
                img[i,j,c] = 0
    img_binary = cv2.
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def straight_line(img,x1,y1,x2,y2):

if __name__=='__main__':
    main()
