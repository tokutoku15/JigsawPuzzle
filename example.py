# -*- coding: utf8 -*-

import cv2
import numpy as np

def main():
    #カラー画像の読み取り
    img_src = cv2.imread("./input/testimg1.png",1)
    
    #グレースケール
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    #二値化
    thresh = 172
    max_value = 255
    _, img_binary = cv2.threshold(img_gray,
                                  thresh,
                                  max_value,
                                  cv2.THRESH_BINARY_INV)

    #ノイズ除去
    kernel = np.ones((20,20),np.uint8)
    dilate = cv2.dilate(img_binary,kernel) #白の領域が膨張
    img_binary = cv2.erode(dilate,kernel) #白の領域が収縮
   
    #ラベリング処理
    #img_binary = cv2.bitwise_not(img_binary)
    Labels = cv2.connectedComponentsWithStats(img_binary)
    #オブジェクト情報を抽出
    LabelNum = Labels[0] - 1
    data = np.delete(Labels[2],0,0)
    center = np.delete(Labels[3],0,0)

    img_color = cv2.cvtColor(img_binary,cv2.COLOR_GRAY2BGR)

    for i in range(LabelNum):
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(img_color,(x0,y0),(x1,y1),(0,0,255))
        cv2.putText(img_color,
                    "ID:"+str(i),
                    (x1-20,y1+15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,(0,255,255))

    showImage(img_color)

    #コーナー検出
    img_corner = cv2.cornerHarris(img_binary,2,3,0.04) 
    img_corner = cv2.dilate(img_corner,None,iterations=5)
    img_binary = cv2.cvtColor(img_binary,cv2.COLOR_GRAY2BGR)
    img_harris = img_binary.copy()
    img_harris = cv2.bitwise_not(img_harris)
    img_harris[img_corner>0.04*img_corner.max()]=[0,0,255]
    showImage(img_harris)

    cv2.imwrite("./output/testimg1_labeling.png",img_color)
    cv2.imwrite("./output/testimg1_harris.png",img_harris)

# 画像の表示関数
def showImage(img):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.waitKey(0)

if __name__=='__main__':
    main()

