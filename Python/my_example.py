import cv2
import numpy as np
import statistics as st
#import matplotlib.pyplot as plt

def main():
    #カラー画像の読み取り
    img_src = cv2.imread("./input/testimg2.png",1)
    
    #画像反転
    img_src = cv2.flip(img_src,1)
    
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
    
    #ラベリング処理
    img_color,data,center = Labeling(img_binary)
    
    removeNoise(img_binary,data,len(data))
    kernel = np.ones((2,2),np.uint8) #カーネルの設定
    dilate = cv2.dilate(img_binary,kernel) #白の領域が膨張
    img_binary = cv2.erode(dilate,kernel) #白の領域が収縮
    #ラベリング
    img_color,data,center = Labeling(img_binary)
    #出力
    #showImage(img_color)
    
    S = 15
    #コーナーの位置指定（"左上":0,"左下":1,"右下":2,"右上":3）
    X_corner = np.zeros((16,4))
    Y_corner = np.zeros((16,4))
    
    for j in range(16):
        img_clip = imgClip(j,data,img_binary)
        img_shi = cv2.cvtColor(img_clip,cv2.COLOR_GRAY2BGR)
        
        X_median = img_shi.shape[0] / 2
        Y_median = img_shi.shape[0] / 2
    
        corners = cv2.goodFeaturesToTrack(img_clip,S,0.01,10)
        corners = np.int0(corners)
        
        count = 0
        X = np.zeros(S)
        Y = np.zeros(S)
        
        for k in range(S):
            X[k] = corners[k][0][0]
            Y[k] = corners[k][0][1]
        
        for i in corners:
            if count <= 2:
                x,y = i.ravel()
                cv2.circle(img_shi,(x,y),3,(0,0,255),-1)
                if count == 2:
                    for l in range(3):
                        if X[l] <= X_median and Y[l] <= Y_median:
                            X_corner[j][0] = X[l]
                            Y_corner[j][0] = Y[l]
                        elif X[l] <= X_median and Y[l] > Y_median:
                            X_corner[j][1] = X[l]
                            Y_corner[j][1] = Y[l]
                        elif X[l] > X_median and Y[l] > Y_median:
                            X_corner[j][2] = X[l]
                            Y_corner[j][2] = Y[l]
                        elif X[l] > X_median and Y[l] <= Y_median:
                            X_corner[j][3] = X[l]
                            Y_corner[j][3] = Y[l]
                    for l in range(4):
                        if X_corner[j][l] == 0:
                            check = l
                            if l == 0:
                                X_th = X_corner[j][l+1]
                                Y_th = Y_corner[j][l+3]
                            elif l == 1:
                                X_th = X_corner[j][l-1]
                                Y_th = Y_corner[j][l+1]
                            elif l == 2:
                                X_th = X_corner[j][l-1]
                                Y_th = Y_corner[j][l+1]
                            else:
                                X_th = X_corner[j][l-1]
                                Y_th = Y_corner[j][l-3]
            else:
                x,y = i.ravel()                
                if X_th - 10 <= x <= X_th + 10 and Y_th - 10 <= y <= Y_th + 10:
                    X_corner[j][check] = X_th
                    Y_corner[j][check] = Y_th                    
                    cv2.circle(img_shi,(x,y),3,(0,0,255),-1)
                    break
            count += 1
        
        #showImage(img_binary)
        cv2.imwrite("./output/testimg"+ str(j) +"_eges.png",img_shi)
        print(X_corner[j],Y_corner[j])

#画像の切り抜き
def imgClip(id,data,src):
    x0 = data[id][0] - 10
    y0 = data[id][1] - 10
    x1 = data[id][0] + data[id][2] + 10
    y1 = data[id][1] + data[id][3] + 10
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
