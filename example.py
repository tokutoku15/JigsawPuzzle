# -*- coding: utf-8 -*-

import cv2
import numpy as np

def main():
    #画像の読み込み
    img = cv2.imread("./input/testimg1.png",1)

    #グレースケール変換
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # showImage(img)

    #二値化
    thresh = 180    #ここの値がピースによって変わる可能性がある
    max_pixel = 255
    ret, img_dst = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    #膨張と縮小でごみ取り
    kernel = np.ones((20,20),np.uint8)
    erode = cv2.erode(img_dst,kernel) #白の領域が収縮
    dilate = cv2.dilate(erode,kernel) #白の領域が膨張
    dilate2 = cv2.dilate(dilate,kernel)
    dst = cv2.erode(dilate2,kernel)

    
    #エッジ検出処理
    canny_img = cv2.Canny(dst,50,200)
    
    # showImage(dst)
    # showImage(canny_img)

    # akaze = cv2.AKAZE_create()
    # kp1 = akaze.detect(img_dst)
    # img_akaze = cv2.drawKeypoints(img_dst,kp1,None,flags=4)

    cv2.imwrite("./output/example_result1.png",dst)
    cv2.imwrite("./output/example_result2.png",canny_img)
    # cv2.imwrite("./output/example_result3.png",img_akaze)

    #辺の特徴検出

    #DB連携

#画像の表示関数
def showImage(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
