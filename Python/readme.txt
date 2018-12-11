imeges		:インプット画像データ6枚
imgs_binary	:二値画像6枚
imgs_data	:短径図形の左上のx,yと縦横の長さと面積
p_data		:104ピース分，ピースデータimgs_dataを連結させた
img_pieces	:ピースの二値画像104枚，listーndarray
p_corners	:104ピース分のコーナー座標，4つ角に値するピース番号には座標(0,0)がある
p_chains	:104ピース分のchainデータ
directions	:8方向ベクトルてきな，座標が(y,x)として入り，
		3 2 1
		4   0
		5 6 7
		に対応
p_points		:104ピース分の輪郭辺の座標,104×(x,y)がN個
p_edge_points_list 	:輪郭辺の座標リスト，416×(x,y)がn個
new_p_corners_list	:コーナー座標のリスト，104×(x,y)が4個
p_rough_list		:左の辺から始めて反時計回りに辺の凹凸情報，104×(-1,0,1)が4つ格納

group1		:4隅のピースのピース番号，4つの値が格納
group2		:輪郭ピースのピース番号，en個の値が格納
group3		:そのほかのピース番号
p_edge_point_conv_list	:p_edge_point_listの並進，回転を加えたもの
