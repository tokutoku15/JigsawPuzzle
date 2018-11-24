import cv2
import numpy as np
import statistics as st
#import matplotlib.pyplot as plt

def main():
    
    """
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
    """
    
    img_binary = cv2.imread("./Python/output/Binary/3_1.png",0)
    
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
    
    Corner(data,img_binary)


#コーナー検出
def Corner(data,img):    
    #試行回数
    S = 6
    #閾値
    size = 20
    #分割数
    split = 20      
    #コーナーの位置指定（"左上":0,"左下":1,"右下":2,"右上":3）
    dst_corner = np.zeros((split,4,2))
    
    for j in range(split):
        img_clip = imgClip(j,data,img)
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
                            dst_corner[j][0][0] = X[l]
                            dst_corner[j][0][1] = Y[l]
                        elif X[l] <= X_median and Y[l] > Y_median:
                            dst_corner[j][1][0] = X[l]
                            dst_corner[j][1][1] = Y[l]
                        elif X[l] > X_median and Y[l] > Y_median:
                            dst_corner[j][2][0] = X[l]
                            dst_corner[j][2][1] = Y[l]
                        elif X[l] > X_median and Y[l] <= Y_median:
                            dst_corner[j][3][0] = X[l]
                            dst_corner[j][3][1] = Y[l]
                    for l in range(4):
                        if dst_corner[j][l][0] == 0:
                            check = l
                            if l == 0:
                                X_th = dst_corner[j][l+1][0] + (dst_corner[j][l+3][0] - dst_corner[j][l+2][0])
                                Y_th = dst_corner[j][l+1][1] + (dst_corner[j][l+3][1] - dst_corner[j][l+2][1])
                            elif l == 1:
                                X_th = dst_corner[j][l-1][0] + (dst_corner[j][l+1][0] - dst_corner[j][l+2][0])
                                Y_th = dst_corner[j][l-1][1] + (dst_corner[j][l+1][1] - dst_corner[j][l+2][1])
                            elif l == 2:
                                X_th = dst_corner[j][l+1][0] + (dst_corner[j][l-1][0] - dst_corner[j][l-2][0])
                                Y_th = dst_corner[j][l+1][1] + (dst_corner[j][l-1][1] - dst_corner[j][l-2][1])
                            else:
                                X_th = dst_corner[j][l-1][0] + (dst_corner[j][l-3][0] - dst_corner[j][l-2][0])
                                Y_th = dst_corner[j][l-1][1] + (dst_corner[j][l-3][1] - dst_corner[j][l-2][1])
            else:
                x,y = i.ravel() 
                cv2.circle(img_shi,(int(X_th),int(Y_th)),3,(255,0,0),-1)
                if X_th - size <= x <= X_th + size and Y_th - size <= y <= Y_th + size:
                    dst_corner[j][check][0] = X_th
                    dst_corner[j][check][1] = Y_th                    
                    cv2.circle(img_shi,(x,y),3,(0,0,255),-1)
            count += 1
        
        #showImage(img_binary)
        cv2.imwrite("./output/test_corner/test"+ str(j) +"_corner.png",img_shi)
    return dst_corner

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
