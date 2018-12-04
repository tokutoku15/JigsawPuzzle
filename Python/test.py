# -*- coding:utf-8 -*-
import math
import numpy as np 
import cv2
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("./output/Piece/piece/1.png")
    # showImage(img)
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img_binary = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # showImage(img_binary)

    p_chains = []
    p_points = [] #pointsリストの作成(x,y)
    #directionリストの作成(x,y)
    directions = [
                [ 1, 0], # 0
                [ 1,-1], # 1
                [ 0,-1], # 2
                [-1,-1], # 3
                [-1, 0], # 4
                [-1, 1], # 5
                [ 0, 1], # 6
                [ 1, 1]  # 7
                ]

    p_chain,p_point = FreemanChainCode(img_binary,directions)
    p_chains.append(p_chain)
    p_points.append(p_point)
    print(p_points[0])
    print(len(p_points[0]))
    print(p_points[0][-10])
    degrees = DegreeEquation(30,p_points[0])

def showImage(img):
    # r_img = cv2.resize(img,(500, 600))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def DegreeEquation(distance,point):
    #対象点の10点前後でcosθを計算する
    degrees = []
    for i in range(-distance,len(point)):
        if i+distance < len(point):
            point_center = point[i]
            point_a = point[i-distance]
            point_b = point[i+distance]
            #回転方向の計算 Rotate>0なら左回りRotate<0なら右回り
            Rotate = (point_a[0]-point_center[0])*(point_b[1]-point_center[1])-(point_b[0]-point_center[0])*(point_a[1]-point_center[1])
            if Rotate < 0:
                Rotate = -1
            else:
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
            denominator = pow(pow(vector_a[0],2)+pow(vector_a[1],2),1/2)*pow(pow(vector_b[0],2)+pow(vector_b[1],2),1/2)   
            numerator = vector_a[0]*vector_b[0]+vector_a[1]*vector_b[1]
            #cosの計算
            cos = numerator/denominator
            theta = math.acos(cos)
            deg = Rotate*math.degrees(theta)
            degrees.append(deg)
            print(deg)

    return degrees
    print("len(deg) = ",len(degrees))
    print("len(points) = ",len(point))


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
    distance_point = (x,y) #最初の地点を記録
    current_point = distance_point
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
                # print(new_point[0],",",new_point[1],"direction=",direction)
                break

    count = 0
    while current_point != distance_point:
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
        if count == 4000: break
        count += 1

    # print(current_point)
    # print(point)
    #print(chain)
    #showImage(src)
    # while current_point != distance_point:
    #     direction = ()
    return chain,point

if __name__ == "__main__":
    main()
