import numpy as np
import cv2
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

    """グレースケール，一度保存できればいい
    for i in range(6):
        img_gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        imname = "./output/Gray/"+ str(i+1) + ".png"
        cv2.imwrite(imname,img_gray)
    """

    #二値化
    #Otsu's thresholding after Gaussian filtering
    imgs_binary = [0,0,0,0,0,0]
    for i in range(6):
        img_gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray,(5,5),0)
        ret3,imgs_binary[i] = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imwrite("./output/Binary/1_"+ str(i+1) + ".png",imgs_binary[i])
    
    #ラベリング処理
    imgs_data = [0,0,0,0,0,0]
    for i in range(6):
        #ラベリング処理(1):大きなノイズもラベリング
        img_color,imgs_data[i] ,center = Labeling(imgs_binary[i])
        #cv2.imwrite("./output/Label/L1_" + str(i+1) + ".png",img_color)
        #ノイズ除去(2):ピース以外のラベリングされた図形の削除
        removeNoise(imgs_binary[i],imgs_data[i],len(imgs_data[i]))
        #cv2.imwrite("./output/Binary/2_" + str(i+1) + ".png",imgs_binary[i])
        img_color,imgs_data[i],center = Labeling(imgs_binary[i])
        #np.savetxt('./output/CSV/before/BeforeData'+ str(i+1) +'.csv',imgs_data[i],delimiter=',')
        #クロージング(3)
        kernel = np.ones((2,2),np.uint8)
        imgs_binary[i] = closing(imgs_binary[i],kernel)
        #cv2.imwrite("./output/Binary/3_" + str(i+1) + ".png",imgs_binary[i])
        #ラベリング処理(3):ノイズ除去済み
        img_color,imgs_data[i],center = Labeling(imgs_binary[i])
        #cv2.imwrite("./output/Label/L2_" + str(i+1) + ".png",img_color)
        #np.savetxt("./output/CSV/after/AfterData"+ str(i+1) +".csv",imgs_data[i],delimiter=',')

    #ピースデータのまとめ
    data = imgs_data[0]
    for i in range(1,6):
        data = np.append(data,imgs_data[i],axis=0)
    """
    np.savetxt("./output/CSV/data.csv",data,delimiter=',')
    np.save('./output/CSV/data.npy', data)
    np.savetxt('./output/Test/binary1.txt',imgs_binary[0])
    np.savetxt("./output/Test/data.csv",data)
    """
    # print(len(imgs_data[5]))
    count = 0
    img_pieces = []
    for a in range(len(images)):
        mimg = cv2.cvtColor(imgs_binary[a],cv2.COLOR_GRAY2BGR)
        count = count + len(imgs_data[a])
        for k in range(count - len(imgs_data[a]),count):
            # print(k)
            img = imgClip(k,data,mimg)
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_pieces.append(img)
    # showImage(img_pieces[0])

    for image in img_pieces:
        # print(image)
        FreemanChainCode(image)
        # print(image.shape[1])
        
# 画像の表示関数
def showImage(img):
    # r_img = cv2.resize(img,(600, 750))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#ラベリング関数
def Labeling(src):
    Labels = cv2.connectedComponentsWithStats(src)
    #オブジェクト情報を抽出
    """
    dataは短径図形の左上のx,yと縦横の長さと面積
    centerは短径図形の重心座標
    LabelNumは領域(背景含む)の個数-1で図形の個数を示す
    deleteしているのは背景分
    """
    LabelNum = Labels[0] - 1
    data = np.delete(Labels[2],0,0)
    center = np.delete(Labels[3],0,0)
    
    dst = cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)
    #dst = src.copy()

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
    #showImage(dst)
    return dst,data,center

#画像の切り抜き
def imgClip(id,data,src):
    x0 = data[id][0] - 5
    y0 = data[id][1] - 5
    x1 = data[id][0] + data[id][2] +5
    y1 = data[id][1] + data[id][3] +5
    if len(src.shape) == 3:
        dst = src[y0:y1,x0:x1,:]
    elif len(src.shape) == 2:
        dst = src[y0:y1,x0:x1]
    return dst

#ノイズ除去関数
def removeNoise(src,data,num):
    id = 0
    for id in range(num):
        img_clip = imgClip(id,data,src)
        if img_clip.size < 10000 :
            x0 = data[id][0]
            y0 = data[id][1]
            x1 = data[id][0] + data[id][2]
            y1 = data[id][1] + data[id][3]
            contours = np.array([[x0,y0],[x0,y1],[x1,y1],[x1,y0]])
            #fillPolyは返り値なしでいい
            cv2.fillPoly(src,pts=[contours],color=(0,0,0))

#クロージングによる黒色ノイズ除去関数
def closing(src,kernel):
    dilate = cv2.dilate(src,kernel) #白の領域が膨張
    dilate = cv2.dilate(dilate,kernel) #白の領域が膨張
    dilate = cv2.erode(dilate,kernel) #白の領域が収縮
    dst = cv2.erode(dilate,kernel) #白の領域が収縮
    return dst

#Freeman chain code 関数
def FreemanChainCode(src):
    #directionリストの作成(y,x)
    directions = [
                [ 0, 1], # 0
                [-1, 1], # 1
                [-1, 0], # 2
                [-1,-1], # 3
                [ 0,-1], # 4
                [ 1,-1], # 5
                [ 1, 0], # 6
                [ 1, 1]  # 7
                ]
    point_is_found = False #見つかった場合True,見つからなかった場合False
    chain = [] #探索した方向を記録していくリスト
    for y in range(src.shape[0]): #y軸走査
        for x in range(src.shape[1]): #x軸走査
            # print(y,x)
            if src[y,x,0] == 255:
                print(1)
                point_is_found = True
                break
        if  point_is_found == True:
            break
    
    # print("(x,y)= (",x,y,")")
    
    start_point = (y,x) #最初の地点を記録
    current_point = start_point
    direction = 2 #最初の点の上にエッジはない
    for i in range(len(directions)):
        if i < direction:
            continue
        else:
            new_point = ( current_point[0]+directions[i][0],
                          current_point[1]+directions[i][1] )
            if src[new_point[0],new_point[1],0] == 255:
                current_point = new_point
                chain.append(i)
                direction = i
                print(new_point[0],",",new_point[1],"direction=",direction)
                break

    count = 0
    while current_point != start_point:
        new_direction = (direction + 5) % 8
        #print("new_direction=",new_direction)
        dir_range1 = range(new_direction,8)
        dir_range2 = range(0,new_direction)
        dirs = []
        dirs.extend(dir_range1)
        dirs.extend(dir_range2)
        for direction in dirs:
            new_point = ( current_point[0]+directions[direction][0],
                          current_point[1]+directions[direction][1] )
            if src[new_point[0],new_point[1],0] == 255:
                chain.append(direction)
                current_point = new_point
                print(new_point[0],",",new_point[1])
                break
        if count == 2000: break
    count += 1

    print(current_point)
    print(chain)
    showImage(src)
    # while current_point != start_point:
    #     direction = ()

    

        

if __name__=='__main__':
    main()
