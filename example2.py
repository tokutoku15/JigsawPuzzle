# -*- coding: utf8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #カラー画像の読み取り
    img_src = cv2.imread("./input/testimg2.png",1)
    
    #画像反転
#    UMat destination;
    img_src = cv2.flip(img_src,1)
#    showImage(img_src)

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

    img_binary = cv2.bitwise_not(img_binary)
#    showImage(img_binary)
    #ラベリング処理
    img_color,data,center = Labeling(img_binary)
    #輪郭抽出
    _,contours, hierarchy = cv2.findContours(cv2.bitwise_not(img_binary),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

    fig, axes = plt.subplots(figsize=(10,10))
    img_contours = cv2.drawContours(img_binary,contours,-1,(0,0,255),3)
#    showImage(img_contours)

    #画像の切り抜き
    removeNoise(img_binary,data,len(data))
    kernel = np.ones((2,2),np.uint8)
    dilate = cv2.dilate(img_binary,kernel) #白の領域が膨張
    img_binary = cv2.erode(dilate,kernel) #白の領域が収縮
    img_color,data,center = Labeling(img_binary)
    showImage(img_color)
    #A-KAZE検出器
#    akaze = cv2.AKAZE_create()
#    kp1, des1 = akaze.detectAndCompute(img_binary,None)
#    kp2, des2 = akaze.detectAndCompute(img_binary2,None)

    #Brute-Force Matcher生成
#    bf = cv2.BFMatcher()

    #特徴量ベクトル同士をBrute-Force&KNNでマッチング
#    matches = bf.knnMatch(des1,des2,k=2)

    # データを間引きする
#    ratio = 0.005
#    good = []
#    for m, n in matches:
#        if m.distance < ratio * n.distance:
#            good.append([m])

    # 対応する特徴点同士を描画
#    img3 = cv2.drawMatchesKnn(img_binary, kp1, img_binary2, kp2, good, None, flags=2)

#    kp2_2 = akaze.detect(img_binary2)
#    img_akaze = cv2.drawKeypoints(img_binary2,kp2_2,None,flags=4)
#    showImage(img_akaze)

    # 画像表示
#    showImage(img_binary2)
#    cv2.imwrite("./output/matching.png",img3)


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


def Labeling(src):
    Labels = cv2.connectedComponentsWithStats(src)
    #オブジェクト情報を抽出
    LabelNum = Labels[0] - 1
    print(LabelNum)
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

def removeNoise(src,data,num):
    id = 0
    for id in range(num):
        img_clip = imgClip(id,data,src)
        print(img_clip.size)
        if img_clip.size < 1000 :
            x0 = data[id][0]
            y0 = data[id][1]
            x1 = data[id][0] + data[id][2]
            y1 = data[id][1] + data[id][3]
            contours = np.array([[x0,y0],[x0,y1],[x1,y1],[x1,y0]])
            cv2.fillPoly(src,pts=[contours],color=(0,0,0))

def draw_contours(axes, img, contours):
    from matplotlib.patches import Polygon
#    axes.imshow(img)
    axes.axis('off')
    for i, cnt in enumerate(contours):
        cnt = np.squeeze(cnt)
        # 点同士を結ぶ線を描画する。
        axes.add_patch(Polygon(cnt, fill=None, lw=2., color='b'))
        # 点を描画する。
        axes.plot(cnt[:, 0], cnt[:, 1],
                  marker='o', ms=4., mfc='red', mew=0., lw=0.)
        # 輪郭の番号を描画する。
        axes.text(cnt[0][0], cnt[0][1], i, color='orange', size='20')

# 画像の表示関数
def showImage(img):
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",img)
    cv2.waitKey(0)

if __name__=='__main__':
    main()
