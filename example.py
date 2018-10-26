# -*- coding: utf-8 -*-

import cv2
import numpy

def main():
    #画像の読み込み
    img = cv2.imread("./input/example.png",1)

    #グレースケール変換
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #二値化
    thresh = 170    #ここの値がピースによって変わる可能性がある
    max_pixel = 255
    ret, img_dst = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    #showImage(img_dst)
    cv2.imwrite("./output/example_result1.png",img_dst)

    #エッジ検出処理
    canny_img = cv2.Canny(img_dst,50,200)
    #showImage(canny_img)
    cv2.imwrite("./output/example_result2.png",canny_img)

    #編のトラッキング

    #DB連携

#画像の表示関数
def showImage(img):
    cv2.imshow("Show Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
