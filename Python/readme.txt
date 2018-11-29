imeges:インプット画像データ6枚
imgs_binary:二値画像6枚
imgs_data:短径図形の左上のx,yと縦横の長さと面積
p_data:104ピース分，ピースデータimgs_dataを連結させた
img_pieces:ピースの二値画像104枚，listーndarray
p_corners:104ピース分のコーナー座標，4つ角に値するピース番号には座標(0,0)がある
p_chains:104ピース分のchainデータ
directions:8方向ベクトルてきな，座標が(y,x)として入り，
		3 2 1
		4   0
		5 6 7
		に対応

