# -*- coding: utf-8 -*-

import cv2
import numpy as np

def main():
    #画像の読み込み
    img = cv2.imread("./input/example.png",1)

    #グレースケール変換
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # showImage(img)

    #二値化
    thresh = 172    #ここの値がピースによって変わる可能性がある
    max_pixel = 255
    _, img_Binary = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    #膨張と縮小とラベリングでごみ取り
    #kernel = np.ones((15,15),np.uint8)
    #erode = cv2.erode(img_Binary,kernel) #白の領域が収縮
    #img_Binary = cv2.dilate(erode,kernel)
    
    #ラベリング処理
    img_Binary = cv2.bitwise_not(img_Binary)
    Labels = cv2.connectedComponentsWithStats(img_Binary)
    #オブジェクト情報を項目別に抽出
    LabelNum = Labels[0] - 1
    data = np.delete(Labels[2],0,0)
    center = np.delete(Labels[3],0,0)
    
    color_src1 = cv2.cvtColor(img_Binary,cv2.COLOR_GRAY2BGR)

    for i in range(LabelNum):
        #各オブジェクトの概説矩形を表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color_src1,(x0,y0),(x1,y1),(0,0,255))
        cv2.putText(color_src1,
                    "ID:"+str(i+1),
                    (x1-20,y1+15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,(0,255,255)) 
        cv2.putText(color_src1,
                     str(int(center[i][0]))+" "+str(int(center[i][1])), 
                    (int(center[i][0]),int(center[i][1])), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1, (0, 255,0))
    
    showImage(color_src1)
#    dilate = cv2.dilate(erode,kernel) #白の領域が膨張

    #エッジ検出処理
    canny_img = cv2.Canny(img_Binary,50,200)

    #辺の特徴検出
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(img_Binary)
    img_akaze = cv2.drawKeypoints(img_Binary,kp1,None,flags=4)
    cv2.imwrite("./output/result.png",img_akaze)
    cv2.imwrite("./output/labeling.png",color_src1)
    #DB連携

#画像の表示関数
def showImage(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
