# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
import numpy as np
from . import generalFunction as genFunc
class Som():

  # テストデータ
  testData = np.array([-100,-5,3,5,10])
  #data形式はこれ（入力データは外部からインプットする）
  #data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]])
  #print(data)

  def __init__(self,node_X,node_Y):
    self.NODE_X = node_X
    self.NODE_Y = node_Y
    self.NODE_K = self.NODE_X * self.NODE_Y

  def runSOM(self,data,count):
    #ノード座標
    node_coordinate_= self._NodeToCoordinate(self.NODE_X,self.NODE_Y)
    #潜在空間の初期化
    #潜在空間の初期化(K X Data次元の乱数の潜在空間を作成）TODO：PCA初期化でもいいよ
    latent_sp_ = np.random.rand(self.NODE_K, len(data[0]))

    # 学習を行う
    count_ =0
    while count_<count:
      # 勝者決定
      win_node_ = self._WinnerNode(data,latent_sp_)
      #print(win_node_)
      # 協調過程
      learn_rate_ = self._CoordinProcess(win_node_,latent_sp_,node_coordinate_,count_)
      # 適応過程
      latent_sp_ = self._AdaptateProcess(learn_rate_,data)
      count_+=1
    #ループ処理おわり

    #勝者ノードの出力
    for n in win_node_:
      print('勝者ノード'+str(n)+':座標')
      print(':座標' +str(node_coordinate_[int(n)]))
    return latent_sp_

  #ノード番号の座標を定義する関数(各ノードの配置は正方形とする)
  #引数　ノードX:int,ノードY:int
  #戻り値 array[ノード数(NodeX＊NodeY),[X座標,Y座標]]
  def _NodeToCoordinate(self,NodeX,NodeY):
    map_tmp = np.zeros((NodeX*NodeY, 2))
    for i in range(NodeY):
      for j in range(NodeX):
        map_tmp[i*NodeX+j]=[i,j]
    return map_tmp

  # 勝者ノードの選定(競合過程)
  # 引数 学習データ、潜在空間
  # 戻り値 勝者ノード：array[データ数]
  def _WinnerNode(self,in_data,latent_sp):
    winner_Kn = np.zeros(len(in_data))
    for indexN,data_n in enumerate(in_data):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm(data_n,latent_sp[0])
      #ノードとデータから最小の差となるノードを選択する
      for index,latent_y in enumerate(latent_sp): #インデックスと値の両方取れる
        tmp = genFunc.Diff2Nolm(data_n,latent_y)
        if dist>tmp:
          dist=tmp
          kn=index
      winner_Kn[indexN]=kn
    return(winner_Kn)

  # 協調過程（ノードと学習データとの学習割合を求める）
  # 引数:勝者ノード、潜在空間、潜在空間の座標(ノード数)、カウント
  # 戻り値　学習割合(全ノード X 入力データ数)
  def _CoordinProcess(self,winner_Kn,latent_sp,node_sp,count):
    # 初期化
    Ykn_Ret = np.zeros((self.NODE_K, len(winner_Kn))) #学習率(潜在空間 X データ数)
    sigma = genFunc.SigmaCalac(count)
  
    for index_k in range(len(latent_sp)):
        for index_N,win_n in enumerate(winner_Kn): #winner_Kn：勝者 
          #参照ベクトル[ノードK,ノードN]＝ガウス（潜在空間ノードkの位置：潜在空間(勝者の位置)、シグマ）
          Ykn_Ret[index_k,index_N]= genFunc.Gauss(node_sp[index_k],node_sp[int(win_n)],sigma)
         #課題ノードをインデックスで管理する　or 配列で整理する
    return Ykn_Ret

  # 適応過程
  # 引数 :学習割合、入力データ
  # 戻り値 :モデルの学習結果(潜在空間：全ノード X データ次元)
  def _AdaptateProcess(self,Ykn_Ret,in_data):
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(self.NODE_K)#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(Ykn_Ret, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
  
    #各ノードの学習割合の標準化
    for index_k in range(self.NODE_K):
      Ykn_Ret[index_k]=Yk_Rec[index_k]*Ykn_Ret[index_k,:]
  
    #参照ノード　✖︎データ
    #以下のリターン文で省略する
    #latent_sp = np.zeros((NODE_K,len(in_data[0])))
    #for index_k in range(NODE_K):
    #    for index_n,data in enumerate(in_data): 
    #      for index_dim,data_dim in enumerate(data):
    #        latent_sp[int(index_k),int(index_dim)] +=Ykn_Ret[int(index_k),int(index_n)]*data_dim
    #return latent_sp  

    return np.dot(Ykn_Ret,in_data)#latent_sp
 