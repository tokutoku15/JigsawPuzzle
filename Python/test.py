# -*- coding:utf-8 -*-
import math
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pickle

def main():
    data = np.loadtxt("./output/CSV/AlltoAll_match.csv",delimiter=",")
    data2 = np.loadtxt("./output/CSV/AlltoAll_match_sy.csv",delimiter=",")
    for i in range(416):
        for j in range(i,416):
            if data[i][j] != data2[i][j]:
                print("OMG")
    
    """
    point_a = loadList("0_0_edge_point_conv")
    point_b = loadList("103_0_edge_point_conv")
    #point_a = [(-2,5),(-3,5),(-3,6),(-4,6),(-4,7),(-5,7),(-5,8)]
    #point_a = [(2,5),(3,5),(3,6),(4,6),(4,7),(5,7),(5,8)]
    #point_a = list(reversed(point_a))
    #point_a = [(3,6),(3,5),(2,5),(2,4),(3,4),(3,3)]
    #point_a = [(0,0),(1,0),(2,1),(3,2),(4,2),(5,3),(5,2),(4,1),(5,0),(6,0),(7,0),(8,0)]
    #point_b = [(0,0),(1,1),(2,0),(3,1),(4,0),(5,1),(6,0)]
    n_a = np.array(point_a)
    n_b = np.array(point_b)
    px = similarityCalc(point_a,point_b)

    point_a_tr = pointTransport(point_a)
    point_a_rot = pointRot(point_a_tr,-1)
    for i in range(len(point_a)):
        print(point_a[i]," >>> ",point_a_tr[i], ">>>" , point_a_rot[i])
    """
    




def showImage(img):
    # r_img = cv2.resize(img,(500, 600))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#リストの保存
def saveList(point_list,name):
    f = open('./output/Point_List/' + str(name) + '.txt', 'wb')
    pickle.dump(point_list, f)

#リストの読み出し
def loadList(name):
    f = open("./output/Point_List/" + str(name) +".txt","rb")
    return pickle.load(f)

def similarityCalc(point_a,point_b):
    #リストを行列へ変換(na×2,nb×2)
    matrix_a = np.matrix(point_a)
    matrix_b = np.matrix(point_b)
    #点数
    na = len(point_a)
    nb = len(point_b)
    #2ノルムの2乗(1×na,1×nb)
    Va = np.matrix(np.diag(np.dot(matrix_a,matrix_a.T)))
    Vb = np.matrix(np.diag(np.dot(matrix_b,matrix_b.T)))
    #要素1のみの行列(1×nb,1×na)
    one_b = np.ones_like(Vb)
    one_a = np.ones_like(Va)
    S = Va.T * one_b - 2*matrix_a*matrix_b.T + one_a.T*Vb
    min_index_a = np.argmin(S,axis=1)
    min_index_b = np.argmin(S,axis=0)
    sa = 0
    sb = 0

    for k in range(S.shape[0]):
        sa = sa + S[k,min_index_a[k,0]]
    for k in range(S.shape[1]):
        sb = sb + S[min_index_b[0,k],k]
    S = sa/na + sb/nb
    S2 = (sa + sb)/(na + nb)
    return S2


#原点に移動
def pointTransport(points):
    dst = []
    dx = points[0][0]
    dy = points[0][1]
    for a in range(len(points)):
        _t = (points[a][0] - dx , points[a][1] - dy)
        dst.append(_t)
    return dst

#回転
#edge_num = 0,1,2,3 , rough = -1,0,1
def pointRot(points,rough):
    dst = []
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    sin = dy / math.sqrt(dx*dx + dy*dy)
    cos = dx / math.sqrt(dx*dx + dy*dy)
    for a in range(len(points)):
        _x = points[a][0]*cos + points[a][1]*sin
        _y = - points[a][0]*sin + points[a][1]*cos
        if rough == -1:
            _y = -_y
        _t = (_x,_y)
        dst.append(_t)
    
    return dst



if __name__ == "__main__":
    main()
