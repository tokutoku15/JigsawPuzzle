# -*- coding:utf-8 -*-
import math
import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def main():
    
    L = np.loadtxt("./output/CSV/data_matrix_L.csv",delimiter=",")
    print(L.shape)
    A = np.array([[3,1],[2,2]])
    la, v = np.linalg.eig(A)

    la, v = np.linalg.eig(L)
    v = np.real(v[:,1:3])
    v2 = np.dot(L,v)
    print(v.shape)

    df = pd.DataFrame(v)
    #x = np.random.randn(5,2)
    #df = pd.DataFrame(x)
    fig, ax = plt.subplots()
    df.plot(0,1,kind='scatter',ax=ax,marker=".",alpha=0.01)
    for k, v in df.iterrows():
        ax.annotate(k//4,xy=(v[0],v[1]),size=8)
    plt.xlim(-0.1,0.1)
    plt.ylim(-0.2,0.2)
    ax.set_xlabel('1')
    ax.set_ylabel('2')

    # save as png
    plt.savefig('./output/test.png')
    plt.close(fig)
    
    """
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(v[0,:],v[1,:])
    ax.set_title('first scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    """
    
    
    """
    data2 = np.loadtxt("./output/CSV/AlltoAll_match_vv.csv",delimiter=",")
    A = np.zeros((12,12))
    print(A)
    s = [0,2,4]
    A[0][s] = [1,1,1]
    I = np.array([[0,1],[1,0]])
    print(I)
    for i in range(0,12,2):
        A[i:i+2,i:i+2] = I
    print(A)
    print(sigmoid(np.array([1,2,3,4])))
    """

    """
    #point_a = loadList("0_0_edge_point_conv")
    #point_b = loadList("10_1_edge_point_conv")
    #point_c = loadList("9_1_edge_point_conv")
    #point_a = [(-2,5),(-3,6),(-4,7),(-5,8),(-6,9),(-6,8),(-6,10)]
    #point_a = [(2,5),(3,6),(4,6),(4,7),(5,7),(5,8)]
    point_a = [(-2,-5),(-3,-6),(-4,-6),(-4,-7),(-5,-7),(-5,-8)]
    
    #point_a = [(3,6),(3,5),(2,5),(2,4),(3,4),(3,3)]
    #point_a = [(0,0),(1,0),(2,1),(3,2),(4,2),(5,1),(6,0)]
    #point_b = [(0,0),(1,1),(2,0),(3,1),(4,0),(5,1),(6,0)]
    #point_a = [(0,0),(0,1),(1,1),(1,0),(2,0)]
    
    point_a = pointTransport(point_a)
    point_a = pointRot(point_a,-1)
    n_a = np.array(point_a)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(n_a[:,0],n_a[:,1])
    ax.set_title('first scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # save as png
    plt.savefig('./output/test.png')
    plt.close(fig)
    #n_b = np.array(point_b)
    #px = similarityCalc(point_a,point_b)
    #py = similarityCalc(point_a,point_c)
    #print(px,"  :  ",py)
    """
    
    """
    for i in range(104):
        for j in range(4):
            point_a = loadList( str(i) +"_" + str(j) + "_edge_point_conv")
            print(len(point_a))
            # figure
            fig = plt.figure()
            n_a = (np.array(point_a)).T
            ax = fig.add_subplot(1, 1, 1)
            # plot
            ax.plot(n_a[0,:], n_a[1,:],'.', color='b', label='y = point')

            # x axis
            ax.set_xlabel('x')
            plt.xlim(-100,800)

            # y axis
            ax.set_ylabel('y')
            plt.ylim(-100,800)

            plt.grid(color='gray')
            # save as png
            plt.savefig('./output/ConvEdge/' + str(i) + '_' + str(j) + '.png')
            plt.close(fig)
    """
            
    
    


def sigmoid(x):
   y = 1 / (1 + np.exp( -x ) )
   return y

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
        _y = -points[a][0]*sin + points[a][1]*cos
        if rough == -1:
            _x = -1*_x
            _y = -1*_y
            _x = math.sqrt(dx*dx + dy*dy) + _x
        _t = (_x,_y)
        dst.append(_t)
    return dst



if __name__ == "__main__":
    main()
