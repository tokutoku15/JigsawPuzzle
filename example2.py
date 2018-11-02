# -*- coding: utf8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #カラー画像の読み取り
    img_src1 = cv2.imread("./input/1T20.png",1)
    img_src2 = cv2.imread("./input/21T40.png",1)
    img_src3 = cv2.imread("./input/41T60.png",1)
    img_src4 = cv2.imread("./input/61T80.png",1)
    img_src5 = cv2.imread("./input/81T96.png",1)
    img_src6 = cv2.imread("./input/97T104.png",1)

    images = [img_src1,img_src2,img_src3,img_src4,img_src5,img_src6]
    #画像反転
#    img_src = cv2.flip(img_src,1)

    #グレースケール
    img_gray = cv2.cvtColor(images[3],cv2.COLOR_BGR2GRAY)
    
    #二値化
    thresh = 172
    max_value = 255
    _, img_binary = cv2.threshold(img_gray,
                                  thresh,
                                  max_value,
                                  cv2.THRESH_BINARY_INV)

    #白黒反転
    img_binary = cv2.bitwise_not(img_binary)
    #ラベリング処理(ノイズ含む)
    img_color,data,center = Labeling(img_binary)
    #ノイズ除去
    removeNoise(img_binary,data,len(data))
    
    kernel = np.ones((2,2),np.uint8)
    dilate = cv2.dilate(img_binary,kernel) #白の領域が膨張
    img_binary = cv2.erode(dilate,kernel) #白の領域が収縮
    #ラベリング処理(ノイズ除去済み)
    img_color,data,center = Labeling(img_binary)
    showImage(img_color)


#画像の切り抜き
def imgClip(id,data,src):
    x0 = data[id][0]
    y0 = data[id][1]
    x1 = data[id][0] + data[id][2]
    y1 = data[id][1] + data[id][3]
    if len(src.shape) == 3:
        dst = src[y0:y1,x0:x1,:]
    elif len(src.shape) == 2:
        dst = src[y0:y1,x0:x1]
    return dst

#ラベリング関数
def Labeling(src):
    Labels = cv2.connectedComponentsWithStats(src)
    #オブジェクト情報を抽出
    LabelNum = Labels[0] - 1
    data = np.delete(Labels[2],0,0)
    center = np.delete(Labels[3],0,0)
    
    dst = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
#    dst = src.copy()

    for i in range(LabelNum):
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(dst,(x0,y0),(x1,y1),(0,0,255))
        cv2.putText(dst,
                    "ID:"+str(i),
                    (x1-20,y1+15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,(0,255,255))
#    showImage(dst)
    return dst,data,center

#ノイズ除去関数
def removeNoise(src,data,num):
    id = 0
    for id in range(num):
        img_clip = imgClip(id,data,src)
        if img_clip.size < 1000 :
            x0 = data[id][0]
            y0 = data[id][1]
            x1 = data[id][0] + data[id][2]
            y1 = data[id][1] + data[id][3]
            contours = np.array([[x0,y0],[x0,y1],[x1,y1],[x1,y0]])
            cv2.fillPoly(src,pts=[contours],color=(0,0,0))

# 画像の表示関数
def showImage(img):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.waitKey(0)

if __name__=='__main__':
    main()
