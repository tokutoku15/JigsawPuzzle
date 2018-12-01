import math
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
    img_num = 6
    p_num = 104
    """グレースケール，一度保存できればいい
    for i in range(6):
        img_gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        imname = "./output/Gray/"+ str(i+1) + ".png"
        cv2.imwrite(imname,img_gray)
    """

    #二値化
    #Otsu's thresholding after Gaussian filtering
    imgs_binary = [0,0,0,0,0,0]
    for i in range(img_num):
        img_gray = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray,(5,5),0)
        ret3,imgs_binary[i] = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imwrite("./output/Binary/1_"+ str(i+1) + ".png",imgs_binary[i])
    
    #ラベリング処理
    imgs_data = [0,0,0,0,0,0]
    for i in range(img_num):
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
    """ピースデータまとめるとき
    p_data = imgs_data[0]
    for i in range(1,img_num):
        p_data = np.append(p_data,imgs_data[i],axis=0)
    np.savetxt("./output/CSV/data.csv",p_data,delimiter=',')
    np.save('./output/CSV/data.npy', p_data)
    np.savetxt('./output/Test/binary1.txt',imgs_binary[0])
    np.savetxt("./output/Test/data.csv",p_data)
    """
    #メモ,27,83,86,100
    #処理が終了したピース数
    count = 0
    #ピースの画像データ104枚を入れる箱
    img_pieces = []
    #ピースのコーナーデータ104つを入れる箱
    p_corners = CornerDetection(imgs_data[0],imgs_binary[0],0,img_pieces)
    count = len(imgs_data[0])
    for i in range(1,len(imgs_data)):
        p_corners = np.append(p_corners,CornerDetection(imgs_data[i],imgs_binary[i],count,img_pieces),axis=0)
        count = count + len(imgs_data[i])
    """
    #ピース番号，象限，x or y
    print(type(p_corners))
    print(p_corners[0])
    print(p_corners[0][1,0])
    """
    p_chains = []
    p_points = [] #pointsリストの作成(x,y)
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
    p_curvature = []
    p_degrees = []
    distance = 50
    for i in range(len(img_pieces)):
        p_chain,p_point = FreemanChainCode(img_pieces[i],directions)
        p_chains.append(p_chain)
        p_points.append(p_point)
        p_curvature.append(calcCurvature(img_pieces[i],p_point))
        degree = DegreeEquation(distance,p_point)
        p_degrees.append(degree)
    
    p_edge_list = []
    p_rough_list = []
    
    for Number in range(len(img_pieces)):
        print("image No.",Number)
        edge_list = corner_dividing(p_corners[Number],p_points[Number],p_degrees[Number])
        p_edge_list.append(edge_list)
        rough_list = judge_roughness(p_edge_list[Number])
        p_rough_list.append(rough_list)
        print("\n")
    """
    #曲率(curvature)とFreemanChainCodeの情報をcsvファイルに出力したいとき．
    p_c_list = []
    for i in range(len(img_pieces)):
        p_c_list.append(p_curvature[i].tolist())
    np.savetxt("./output/CSV/curvature.csv", p_c_list, fmt='%s',delimiter=',')
    np.savetxt("./output/CSV/FreemanChainCode.csv", p_chains, fmt='%s', delimiter=',')
    """
    #showImage(img_pieces[0])


# 画像の表示関数
def showImage(img):
    r_img = cv2.resize(img,(500, 600))
    cv2.imshow("img",r_img)
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
    x0 = data[id][0] - 15
    y0 = data[id][1] - 15
    x1 = data[id][0] + data[id][2] + 15
    y1 = data[id][1] + data[id][3] + 15
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

#コーナー検出と各ピースの保存
#短形データ，二値画像，現在までの処理を終えたピースの番号，ピース画像格納庫
def CornerDetection(data,img,pnum,img_pieces):    
    #goodFeaturesToTrackの試行回数
    S = 13
    #閾値
    size = 20
    #分割数
    split = len(data)
    #コーナーの位置指定，二次元座標系の象限（"左上":0,"左下":1,"右下":2,"右上":3）
    dst_corner = np.zeros((split,4,2))
    
    for j in range(split):
        img_clip = imgClip(j,data,img)
        img_pieces.append(img_clip)
        img_shi = cv2.cvtColor(img_clip,cv2.COLOR_GRAY2BGR)
        
        X_median = img_shi.shape[0] / 2
        Y_median = img_shi.shape[0] / 2
    
        corners = cv2.goodFeaturesToTrack(img_clip,S,0.01,10)
        corners = np.int0(corners)
        
        count = 0
        X = np.zeros(S,dtype=np.int64)
        Y = np.zeros(S,dtype=np.int64)
        
        for k in range(S):
            X[k] = corners[k][0][0]
            Y[k] = corners[k][0][1]
        d_min = size
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
                d = math.sqrt(pow(X_th - x,2) + pow(Y_th - y,2))
                if d < d_min:
                    d_min = d
                    dst_corner[j][check][0] = x
                    dst_corner[j][check][1] = y
            count += 1
        cv2.circle(img_shi,(int(dst_corner[j][check][0]),int(dst_corner[j][check][1])),3,(0,0,255),-1)
        cv2.imwrite("./output/Piece/Corner/"+ str(pnum + j) +"_corner.png",img_shi)
    return dst_corner



#Freeman chain code 関数
def FreemanChainCode(src,directions):
    point_is_found = False #見つかった場合True,見つからなかった場合False
    chain = [] #探索した方向を記録していくリスト
    point = [] #探索した座標を記録していくリスト
    for y in range(src.shape[0]): #y軸走査
        for x in range(src.shape[1]): #x軸走査
            # print(y,x)
            if src[y,x] == 255:
                #print(1)
                point_is_found = True
                break
        if  point_is_found == True:
            break
    # print("(x,y)= (",x,y,")")
    start_point = (x,y) #最初の地点を記録
    current_point = start_point
    direction = 2 #最初の点の上にエッジはない
    for i in range(len(directions)):
        if i < direction:
            continue
        else:
            new_point = ( current_point[0]+directions[i][0],
                          current_point[1]+directions[i][1] )
            if src[new_point[1],new_point[0]] == 255:
                current_point = new_point
                point.append(current_point)
                chain.append(i)
                direction = i
                #print(new_point[0],",",new_point[1],"direction=",direction)
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
            if src[new_point[1],new_point[0]] == 255:
                chain.append(direction)
                current_point = new_point
                point.append(current_point)
                #print(new_point[0],",",new_point[1])
                break
        if count == 2000: break
        count += 1
    # print(current_point)
    # print(point)
    #print(chain)
    #showImage(src)
    # while current_point != start_point:
    #     direction = ()
    return chain,point


def DegreeEquation(distance,point):
    #対象点の10点前後でcosθを計算する
    degree = []
    point_temp = []
    for i in range(2):
        point_temp.extend(point)
    # print(point_temp)
    # print(len(point))
    for i in range(len(point)):
   
        point_center = point_temp[i]
        point_a = point_temp[i-distance]
        point_b = point_temp[i+distance]
        #回転方向の計算 Rotate>0なら左回りRotate<0なら右回り
        Rotate = (point_a[0]-point_center[0])*(point_b[1]-point_center[1])-(point_b[0]-point_center[0])*(point_a[1]-point_center[1])
        if Rotate < 0:
            Rotate = -1
        elif Rotate > 0:
            Rotate = 1

        x = point_center[0]
        y = point_center[1]
        #ベクトルa->,b->を計算
        vector_a = (point_a[0]-x,
                point_a[1]-y)
        vector_b = (point_b[0]-x,
                point_b[1]-y)
        #内積の計算
        #分母
        denominator = pow(pow(vector_a[0],2)+pow(vector_a[1],2),0.5)*pow(pow(vector_b[0],2)+pow(vector_b[1],2),0.5)   
        #分子
        numerator = vector_a[0]*vector_b[0]+vector_a[1]*vector_b[1]
        #cosの計算
        cos = clean_cos(numerator/denominator)
        theta = math.acos(cos)
        deg = Rotate*math.degrees(theta)
        degree.append(deg)
        # print("center = ",point_center," a = ",point_a," b = ",point_b)
        # print("cos = ",cos,"theta = ",theta,"deg = ",deg,"degree=",degree[i])
        # print("denominator=",denominator,",numerator=",numerator,"\n")
    # print("len(deg) = ",len(degrees))
    # print("len(points) = ",len(point))
    # print(degree)
    return degree

# DegreeEquationのacos外れ値を定義内に調整する関数
def clean_cos(cos_angle): 
    return min(1,max(cos_angle,-1)) 

#コーナー間で辺の情報を分ける関数
def corner_dividing(corners,points,degrees):
    #cornersがnparray型なのでリストに変換
    corners_list = []
    corners_num = len(corners)
    for i in range(corners_num):
        corners_list.append((corners[i][0],corners[i][1]))
    # print(corners_list)
    #(0,0)の点をリストから削除する
    zero = (0,0)
    if zero in corners_list:
        corners_list.remove(zero)
    corners_num = len(corners_list)
    
    # if len(corners_list) == 3:
    #     print("len(corners) = 3")
    # elif len(corners_list) == 4:
    #     print("len(corners) = 4")
    # else:
    #     print("len(corners) = ",len(corners_list))

    #相似度を計算してコーナー点の近似点を探す
    point_num = len(points)
    for i in range(corners_num):
        if corners_list[i] in points:
            continue
        else:
            subtract = ( points[0][0]-corners_list[i][0],
                         points[0][1]-corners_list[i][1] )
            min_similarity = pow(subtract[0]*subtract[0]
                                +subtract[1]*subtract[1] , 1/2)
            near_point = points[0]
            for j in range(1,point_num):
                subtract = ( points[j][0]-corners_list[i][0],
                             points[j][1]-corners_list[i][1] )
                simmilarity = pow(subtract[0]*subtract[0]
                                 +subtract[1]*subtract[1] , 1/2)
                if min_similarity > simmilarity:
                    min_similarity = simmilarity
                    near_point = points[j]
            corners_list[i] = near_point
    #デバッグ用
    corner_index_list = []
    for i in range(len(corners_list)):
        if corners_list[i] in points:
            # print(corners_list[i],"in points[",points.index(corners_list[i]),"]")
            corner_index_list.append(points.index(corners_list[i]))
    #コーナー間の辺を分割・記録
    record_index = min(corner_index_list)
    temp_record_index = corner_index_list.index(record_index)
    # print("temp = ",temp_record_index)
    # print("points_length = ",point_num)
    edge = []
    edge_list = []
    print("corner_index_list = ",corner_index_list)
    for i in range(len(corner_index_list)):
        print("corner:",i," = ",points[corner_index_list[i]])
    for i in range(point_num):
        index = (i+record_index)%point_num
        if i < point_num-1:
            if index != corner_index_list[(temp_record_index+1)%len(corner_index_list)]:
                edge.append(degrees[index])
            elif index == corner_index_list[(temp_record_index+1)%len(corner_index_list)]:
                # print(edge)
                # print("len(edge)=",len(edge))
                edge_list.append(edge[:])
                # print("edge_list[0]=",edge_list[0])
                edge.clear()
                temp_record_index = (temp_record_index+1)%len(corner_index_list)
                edge.append(degrees[index])
        else:
            edge.append(degrees[index])
            edge_list.append(edge[:])
            # print(edge)
            # print("len(edge)=",len(edge))
    # print("process finish.")
    # print(edge_list)
    return edge_list

#辺それぞれが凹凸か判定する関数
def judge_roughness(edge_list):
    '''
    roughリストの作成
    1...凸/ -1...凹/ 0...直線
    '''
    rough_list = []
    for i in range(len(edge_list)):
        # print(edge_list[i])
        print(edge_list[i][len(edge_list[i])//2])
        if abs(edge_list[i][len(edge_list[i])//2]) < 170:
            if edge_list[i][len(edge_list[i])//2] > 10:
                print("凸")
                rough_list.append(1)
            elif edge_list[i][len(edge_list[i])//2] < -10:
                print("凹")
                rough_list.append(-1)
            else:
                print("直線")
                rough_list.append(0)
        else:
            print("直線")
            rough_list.append(0)
    return rough_list



def calcCurvature(img_b,p_point):
    #ガウシアン
    img = cv2.GaussianBlur(img_b, ksize=(5,5), sigmaX=1.3)
    #水平の微分(縦方向の検出)
    xkernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    fx = cv2.filter2D(img, -1, xkernel)
    fxx = cv2.filter2D(fx, -1, xkernel)
    #垂直(鉛直方向の検出)
    ykernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    fy = cv2.filter2D(img, -1, ykernel)
    fyy = cv2.filter2D(fy, -1, ykernel)
    #斜
    fxy = cv2.filter2D(fx, -1, ykernel)
    fyx = cv2.filter2D(fy, -1, xkernel)
    fxyyx = cv2.add(fxy / 2 , fyx / 2)
    dst = np.zeros(len(p_point))
    j = 0
    for i in p_point:
        #cは座標を転置して参照しやすいように
        c = (i[1],i[0])
        numerator = int(fxx[c]) + int(fyy[c]) + int(fxx[c])*math.pow(fy[c],2.0) + int(fyy[c])*math.pow(fx[c],2.0) - 2*int(fx[c])*int(fy[c])*int(fxyyx[c])
        denominator = float(2*(1 + math.pow(fx[c],2.0) + math.pow(fy[c],2.0)))
        dst[j] = round(numerator / denominator , 8)
        j = j+1
    return dst




if __name__=='__main__':
    main()