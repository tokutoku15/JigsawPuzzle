# -*- coding:utf-8 -*-
import math
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pickle

def main():
    
    data2 = np.loadtxt("./output/CSV/AlltoAll_match_vv.csv",delimiter=",")
    sim5 = []
    
    #候補cand番目まで
    cand = 5
    for i in range(data2.shape[0]):
        sort_index = np.argsort(data2[i])
        sort_index2 = []
        c = 0
        for j in range(len(sort_index)):
            if (data2[i][sort_index[j]] > 0.000001):
                c = c + 1
                sort_index2.append(sort_index[j])
            if c >= cand:
                break
        sort_index2 = (np.array(sort_index2))
        sim5.append(sort_index2)
        if(p_rough_list[i//4][i%4] != 0):
            print(i , " : " , sim5[i])
            print(i , " : " , data2[i][sim5[i]])
    #(piece num , edge num)
    print(data2[415][sort_index])
    match_list = []
    
    for i in range(len(sim5)):
        k = []
        #cand個の(piece_num,edge_num)を生成
        if(p_rough_list[i//4][i%4] != 0):
            for j in range(cand):
                p_num = sim5[i][j] // 4
                e_num = sim5[i][j] % 4
                k.append((p_num,e_num))
        else:
            k.append(())
        match_list.append(k)
        print((i//4,i%4)," >>> ",k)
        if(sim5[i] != ()):
            print((i//4 , i%4) , " : " , data2[i][sim5[i]])
        else:
            print((i//4,i%4)," >>> Line")
        print("\n")

    """
    
    point_a = loadList("0_0_edge_point_conv")
    point_b = loadList("10_1_edge_point_conv")
    point_c = loadList("9_1_edge_point_conv")
    #point_a = [(-2,5),(-3,5),(-3,6),(-4,6),(-4,7),(-5,7),(-5,8)]
    #point_a = [(2,5),(3,5),(3,6),(4,6),(4,7),(5,7),(5,8)]
    #point_a = list(reversed(point_a))
    #point_a = [(3,6),(3,5),(2,5),(2,4),(3,4),(3,3)]
    #point_a = [(0,0),(1,0),(2,1),(3,2),(4,2),(5,3),(5,2),(4,1),(5,0),(6,0),(7,0),(8,0)]
    #point_b = [(0,0),(1,1),(2,0),(3,1),(4,0),(5,1),(6,0)]
    n_a = np.array(point_a)
    n_b = np.array(point_b)
    px = similarityCalc(point_a,point_b)
    py = similarityCalc(point_a,point_c)
    print(px,"  :  ",py)
    
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

#類似度計算
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
    th = 1
    pen = 10
    for k in range(S.shape[0]):
        v = S[k,min_index_a[k,0]]
        if(v > th):
            v = v*v
        elif(v < th):
            v = v*(1/pen)
        sa = sa + v
    for k in range(S.shape[1]):
        v = S[min_index_b[0,k],k]
        if(v > th):
            v = v*v
        elif(v < th):
            v = v*(1/pen)
        sb = sb + v
    S2 = sa/na + sb/nb
    
    for i in range(len(point_a)):
        print(point_a[i]," >>> ",point_b[min_index_a[i,0]]," : ",S[i,min_index_a[i,0]])
    
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
