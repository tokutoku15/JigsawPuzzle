import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    c = 0
    data = np.load('./output/CSV/data.npy')
    for a in range(4):
        mimg = cv2.imread('./output/Binary/3_' + str(a+1) + '.png')
        for k in range(20):
            print(k)
            img = imgClip(c,data,mimg)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
            print(type(corners))
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(img,(x,y),3,(0,0,255),-1)
            cv2.imwrite("./output/Test/a/" + str(c) + ".jpg", img)
            c = c+1
    """
    filename = './output/Binary/3_1.png'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.049)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imwrite("./output/Test/a2.jpg", img)
    """
    """
    img = np.loadtxt('./output/Test/binary1.txt', np.uint8)
    gray = np.float32(img)
    """
    """
    #カラー画像の読み取り
    img = cv2.imread("./output/Binary/3_1.png",1)
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 方法1(NumPyで実装)
    edge1 = canny_edge_detecter(gray, 100, 200, 1)
    # 結果を出力
    cv2.imwrite("./output/Test/canny.jpg", edge1)
    """
    """
    img_test = cv2.imread("./output/Binary/3_1.png",0)
    print(type(img_test))
    print("s")
    imgss = ZhangSuen(img_test)
    np.save('./output/Test/sample_1.npy', imgss)
    cv2.imwrite("./output/Test/imgss.png",imgss)
    """
    """
    img_src1 = cv2.imread("./input/1T20.png",1)
    img_src2 = cv2.imread("./input/21T40.png",1)
    img_src3 = cv2.imread("./input/41T60.png",1)
    img_src4 = cv2.imread("./input/61T80.png",1)
    img_src5 = cv2.imread("./input/81T96.png",1)
    img_src6 = cv2.imread("./input/97T104.png",1)
    images = [img_src1,img_src2,img_src3,img_src4,img_src5,img_src6]
    #グレースケール
    img_gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    imname = "./output/Test/a.png"
    cv2.imwrite(imname,img_gray)

    count = 0
    jc = 0
    height, width = img_gray.shape[:2]
    for i in range(height):
        for j in range(width):
            if(img_test[i,j] == img_gray[i,j]):
                count = count + 1
            if(j+1 == width):
                jc = jc + 1
    print(str(count) + " size" + str(height*width))
    print(str(jc) + "  height " + str(height))
    """
    
    """大津二値化
    img_gray = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(str(ret2))
    cv2.imwrite("./output/Test/OtsuTh1.png",th2)
    #Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("./output/Test/OtsuTh2.png",th3)
    """
    
    

                
def fun(src):
    contours = np.array([[0,0],[0,200],[200,200],[200,0]])
    cv2.fillPoly(src,pts=[contours],color=(0,0,0))



# 画像の表示関数
def showImage(img):
    r_img = cv2.resize(img,(600, 750))
    cv2.imshow("img",r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#画像の切り抜き
def imgClip(id,data,src):
    x0 = data[id][0] - 1
    y0 = data[id][1] - 1
    x1 = data[id][0] + data[id][2] +1
    y1 = data[id][1] + data[id][3] +5
    if len(src.shape) == 3:
        dst = src[y0:y1,x0:x1,:]
    elif len(src.shape) == 2:
        dst = src[y0:y1,x0:x1]
    return dst

# 与えられた引数全てについての論理積を返すメソッドです。
def multi_logical_and(*args):
    result = np.copy(args[0])
    for arg in args:
        result = np.logical_and(result, arg)
    return result

# 2値画像について、周囲1ピクセルをFalseで埋めるメソッドです
def padding(binary_image):
    row, col = np.shape(binary_image)
    result = np.zeros((row+2,col+2))
    result[1:-1, 1:-1] = binary_image[:, :]
    return result

# paddingの逆です
def unpadding(image):
    image = image[1:-1, 1:-1]
    h,w = image.shape
    dst = np.zeros((h,w), dtype='uint8')
    for i in range(h):
        for j in range(w):
            if(image[i,j]):
                dst[i,j] = int(255)
            else:
                dst[i,j] = int(0)
    return dst

# そのピクセルの周囲のピクセルの情報を格納したarrayを返します。
def generate_mask(image):
    row, col = np.shape(image)
    p2 = np.zeros((row, col)).astype(bool)
    p3 = np.zeros((row, col)).astype(bool)
    p4 = np.zeros((row, col)).astype(bool)
    p5 = np.zeros((row, col)).astype(bool)
    p6 = np.zeros((row, col)).astype(bool)
    p7 = np.zeros((row, col)).astype(bool)
    p8 = np.zeros((row, col)).astype(bool)
    p9 = np.zeros((row, col)).astype(bool)
    #上
    p2[1:row-1, 1:col-1] = image[0:row-2, 1:col-1]
    #右上
    p3[1:row-1, 1:col-1] = image[0:row-2, 2:col]
    #右
    p4[1:row-1, 1:col-1] = image[1:row-1, 2:col]
    #右下
    p5[1:row-1, 1:col-1] = image[2:row, 2:col]
    #下
    p6[1:row-1, 1:col-1] = image[2:row, 1:col-1]
    #左下
    p7[1:row-1, 1:col-1] = image[2:row, 0:col-2]
    #左
    p8[1:row-1, 1:col-1] = image[1:row-1, 0:col-2]
    #左上
    p9[1:row-1, 1:col-1] = image[0:row-2, 0:col-2]
    return (p2, p3, p4, p5, p6, p7, p8, p9)

# 周囲のピクセルを順番に並べたときに白→黒がちょうど1箇所だけあるかどうかを判定するメソッドです。
def is_once_change(p_tuple):
    number_change = np.zeros_like(p_tuple[0])
    # P2~P9,P2について、隣接する要素の排他的論理和を取った場合のTrueの個数を数えます。
    for i in range(len(p_tuple) - 1):
        number_change = np.add(number_change, np.logical_xor(p_tuple[i], p_tuple[i+1]).astype(int))
    number_change = np.add(number_change, np.logical_xor(p_tuple[7], p_tuple[0]).astype(int))
    array_two = np.ones_like(p_tuple[0]) * 2

    return np.equal(number_change, array_two)

# 周囲の黒ピクセルの数を数え、2以上6以下となっているかを判定するメソッドです。
def is_black_pixels_appropriate(p_tuple):
    number_of_black_pxels = np.zeros_like(p_tuple[0])
    array_two = np.ones_like(p_tuple[0]) * 2
    array_six = np.ones_like(p_tuple[0]) * 6
    for p in p_tuple:
        number_of_black_pxels = np.add(number_of_black_pxels, p.astype(int))
    greater_two = np.greater_equal(number_of_black_pxels, array_two)
    less_six = np.less_equal(number_of_black_pxels, array_six)
    return np.logical_and(greater_two, less_six)

def step1(image, p_tuple):
    #条件1
    condition1 = np.copy(image)

    #条件2
    condition2 = is_once_change(p_tuple)

    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)

    #条件4
    condition4 = np.logical_not(multi_logical_and(p_tuple[0], p_tuple[2], p_tuple[4]))

    #条件5
    condition5 = np.logical_not(multi_logical_and(p_tuple[2], p_tuple[4], p_tuple[6]))

    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

def step2(image, p_tuple):
    #条件1
    condition1 = np.copy(image)
    #条件2
    condition2 = is_once_change(p_tuple)
    #条件3
    condition3 = is_black_pixels_appropriate(p_tuple)
    #条件4
    condition4 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[2], p_tuple[6])))
    #条件5
    condition5 = np.logical_not(np.logical_and(p_tuple[0], np.logical_and(p_tuple[4], p_tuple[6])))
    return np.logical_xor(multi_logical_and(condition1, condition2, condition3, condition4, condition5), image)

# 2値化画像を細線化して返すメソッドです。
def ZhangSuen(image):

    image = padding(image)

    while True:
        old_image = np.copy(image)

        p_tuple = generate_mask(image)
        image = step1(image, p_tuple)
        p_tuple = generate_mask(image)        
        image = step2(image, p_tuple)

        if (np.array_equal(old_image, image)):
            break

    return unpadding(image)

# 畳み込み演算（空間フィルタリング）
def filter2d(src, kernel, fill_value=-1):
    # カーネルサイズ
    m, n = kernel.shape
    
    # 畳み込み演算をしない領域の幅
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]
    
    # 出力画像用の配列
    if fill_value == -1: dst = src.copy()
    elif fill_value == 0: dst = np.zeros((h, w))
    else:
        dst = np.zeros((h, w))
        dst.fill(fill_value)
    
    for y in range(d, h - d):
        for x in range(d, w - d):
            # 畳み込み演算
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)
            
    return dst


# Non maximum Suppression処理
def non_max_sup(G, Gth):

    h, w = G.shape
    dst = G.copy()

    # 勾配方向を4方向(垂直・水平・斜め右上・斜め左上)に近似
    Gth[np.where((Gth >= -22.5) & (Gth < 22.5))] = 0
    Gth[np.where((Gth >= 157.5 ) & (Gth < 180))] = 0
    Gth[np.where((Gth >= -180 ) & (Gth < -157.5))] = 0
    Gth[np.where((Gth >= 22.5) & (Gth < 67.5))] = 45
    Gth[np.where((Gth >= -157.5 ) & (Gth < -112.5))] = 45
    Gth[np.where((Gth >= 67.5) & (Gth < 112.5))] = 90
    Gth[np.where((Gth >= -112.5) & (Gth < -67.5))] = 90
    Gth[np.where((Gth >= 112.5) & (Gth < 157.5))] = 135
    Gth[np.where((Gth >= -67.5) & (Gth < -22.5))] = 135

    # 注目画素と勾配方向に隣接する2つの画素値を比較し、注目画素値が最大でなければ0に
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if Gth[y][x]==0:
                if (G[y][x] < G[y][x+1]) or (G[y][x] < G[y][x-1]):
                    dst[y][x] = 0
            elif Gth[y][x] == 45:
                if (G[y][x] < G[y-1][x+1]) or (G[y][x] < G[y+1][x-1]):
                    dst[y][x] = 0
            elif Gth[y][x] == 90:
                if (G[y][x] < G[y+1][x]) or (G[y][x] < G[y-1][x]):
                    dst[y][x] = 0
            else:
                if (G[y][x] < G[y+1][x+1]) or  (G[y][x] < G[y-1][x-1]):
                    dst[y][x] = 0
    return dst


# Hysteresis Threshold処理
def hysteresis_threshold(src, t_min=75, t_max=150, d=1):

    h, w = src.shape
    dst = src.copy()

    for y in range(0, h):
        for x in range(0, w):
            # 最大閾値より大きければ信頼性の高い輪郭
            if src[y][x] >= t_max: dst[y][x] = 255
            # 最小閾値より小さければ信頼性の低い輪郭(除去)
            elif src[y][x] < t_min: dst[y][x] = 0
            # 最小閾値～最大閾値の間なら、近傍に信頼性の高い輪郭が1つでもあれば輪郭と判定、無ければ除去
            else:
                if np.max(src[y-d:y+d+1, x-d:x+d+1]) >= t_max:
                    dst[y][x] = 255
                else: dst[y][x] = 0

    return dst


def canny_edge_detecter(gray, t_min, t_max, d):

    """
    # 処理1 ガウシアンフィルタで平滑化      
    kernel_g = np.array([[1/16, 1/8, 1/16],
                         [1/8,  1/4,  1/8],
                         [1/16, 1/8, 1/16]])

    # ガウシアンフィルタ
    G = filter2d(gray, kernel_g, -1)
    """
    G = gray.copy()
    # 処理2 微分画像の作成（Sobelフィルタ）
    kernel_sx = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
    kernel_sy =  np.array([[-1,-2,-1],
                           [0,  0, 0],
                           [1,  2, 1]])
    Gx = filter2d(G, kernel_sx, 0)
    Gy = filter2d(G, kernel_sy, 0)
    
    # 処理3 勾配強度・方向を算出
    G = np.sqrt(Gx**2 + Gy**2)
    Gth = np.arctan2(Gy, Gx) * 180 / np.pi

    # 処理4 Non maximum Suppression処理
    G = non_max_sup(G, Gth)

    # 処理5 Hysteresis Threshold処理
    return hysteresis_threshold(G, t_min, t_max, d)


if __name__=='__main__':
    main()