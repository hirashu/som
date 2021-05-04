# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
#今回の学習データData X K X L X 1 であると仮定する
import numpy as np
from . import generalFunction as genFunc
class TSom():

  def __init__(self,node_KX,node_KY,node_LX,node_LY):
      # ノード設定
      #NODE_K・Lとする
    self.NODE_KX = node_KX
    self.NODE_KY = node_KY
    self.NODE_K = self.NODE_KX * self.NODE_KY
    self.NODE_LX = node_LX
    self.NODE_LY = node_LY
    self.NODE_L = self.NODE_LX * self.NODE_LY

  #属性情報を含めたTSOM
  #sideData サイド情報はN*Nの正方行列
  #前提：学習データは次元数１である。
  #   ：属性情報の次元数は未設定
  # 関数化（入力データ）
  #data形式は以下の形式
  #data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]])
  #return　計算結果(配列)
  def runTSOM2SideInfo(self,data,sideData):
    #ノード座標
    nodeK_coordinate_=self.NodeToCoordinate(self.NODE_KX,self.NODE_KY)
    nodeL_coordinate_=self.NodeToCoordinate(self.NODE_LX,self.NODE_LY)
    #潜在空間の参照ベクトル初期化(TODO：PCA初期化でやろう)
    latent_spU1_ = np.random.rand(len(data),self.NODE_L) #memoデータ数
    latent_spU2_ = np.random.rand(self.NODE_K,len(data[0])) #memo次元のはず
    #本来はL*N*Dであるがデータの次元が１であるためN*Lに省略
    latent_spY_ = np.random.rand(self.NODE_K,self.NODE_L)
    #print('latent_spY_')
    #print(latent_spY_)
    #属性情報の潜在空間(R ∈ Node * D)を定義
    lspYsideK_ = np.random.rand(self.NODE_K,len(sideData[0]))
    lspYsideL_ = np.random.rand(self.NODE_L,len(sideData[0]))

    #学習を行う
    count_ =0
    while count_<250:
      # 勝者決定
      win_nodeK_ = self.WinnerNodeK_sideInfo(latent_spU1_,latent_spY_,sideData,lspYsideK_)
      #print('win_nodeK_')
      #print(win_nodeK_)
      win_nodeL_ = self.WinnerNodeL_sideInfo(latent_spU2_,latent_spY_,sideData,lspYsideL_)
      #print('win_nodeL_')
      #print(win_nodeL_)
      #print(win_node_)
      # 協調過程
      learn_rate_K =self.CoordinProcess(win_nodeK_,self.NODE_K,nodeK_coordinate_,count_)
      #print('learn_rate_K')
      #print(learn_rate_K)

      learn_rate_L =self.CoordinProcess(win_nodeL_,self.NODE_L,nodeL_coordinate_,count_)
      #print('learn_rate_L')
      #print(learn_rate_L)

      # 適応過程
      #各ノードの学習率を求める
      learn_rate_K=self.LearnStandardization(learn_rate_K)
      learn_rate_L=self.LearnStandardization(learn_rate_L)
      #潜在空間の更新
      latent_spU1_ = np.dot(data,learn_rate_L.T)
      #print('latent_spU1_')
      #print(latent_spU1_)
      #latent_spU2_ = np.dot(learn_rate_K,data)
      latent_spU2_=self.AdaptateProcessU2(learn_rate_K,data)
      #print('latent_spU2_')
      #print(latent_spU2_)
      latent_spY_= np.dot(learn_rate_K,latent_spU1_)

      #潜在空間（属性情報）の更新
      lspYsideK_= np.dot(learn_rate_K,sideData)
      lspYsideL_= np.dot(learn_rate_L,sideData)
      count_+=1
      print(count_)
    #ループ処理おわり

    #学習結果の出力
    #for n in win_nodeK_:
    #  print('勝者ノード'+str(n)+':座標')
    #  print(':座標' +str(nodeK_coordinate_[int(n)]))
    print('win_nodeK_縦方向')
    print(win_nodeK_)
    print('win_nodeL_横方向')
    print(win_nodeL_)
    return latent_spY_

  #ノード番号の座標を定義する関数(各ノードの配置は正方形とする)
  #引数　ノードX:int,ノードY:int
  #戻り値 array[ノード数(NodeX＊NodeY),[X座標,Y座標]]
  def NodeToCoordinate(self,NodeX,NodeY):
    map_tmp = np.zeros((NodeX*NodeY, 2))
    for i in range(NodeY):
      for j in range(NodeX):
        map_tmp[i*NodeX+j]=[i,j]
    return map_tmp

  # 勝者ノード(第１ノード)の選定(競合過程)
  # 引数 U1(N*L)、潜在空間Y(K*L)
  # 戻り値 勝者ノード：array[データ数N]
  def WinnerNodeK(self,latent_spU1,latent_spY):
    winner_Kn = np.zeros(len(latent_spU1)) #len=n1
    #データ数分勝者の算出を繰り返す
    for indexN,data_u1 in enumerate(latent_spU1):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm(data_u1,latent_spY[0])
      #Lノード分やるよー
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K): #Lノード回の中からそれっぽいのを決める
        tmp = genFunc.Diff2Nolm(data_u1,latent_spY[index_k])
        if dist>tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn
    return(winner_Kn)

  # 勝者ノード(第２ノード)の選定(競合過程)
  # 引数 学習データ(L*Data数)、潜在空間
  # 戻り値 勝者ノード：array[データ次元数]
  def WinnerNodeL(self,latent_spU2,latent_spY):
    winner_Ln = np.zeros(len(latent_spU2[0]))
    latent_spU2T=latent_spU2.T
    latent_spYT=latent_spY.T
    for indexN,data_n in enumerate(latent_spU2T):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm(data_n,latent_spYT[0])
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L): #インデックスと値の両方取れる
        tmp = genFunc.Diff2Nolm(data_n,latent_spYT[index_l])
        if dist>tmp:
          dist=tmp
          kn=index_l
      winner_Ln[indexN]=kn
    return(winner_Ln)

  # 勝者ノード(第１ノード)の選定_属性情報を含む(競合過程)
  # 引数 U1(N*L)、潜在空間Y(K*L)、属性情報(N*D)、属性情報の潜在空間(K*D)
  # 戻り値 勝者ノード：array[データ数N]
  def WinnerNodeK_sideInfo(self,latent_spU1,latent_spY,input_Data,lspYsideK):
    winner_Kn = np.zeros(len(latent_spU1)) #len=n1
    #データ数分勝者の算出を繰り返す
    for indexN,data_u1 in enumerate(latent_spU1):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm(data_u1,latent_spY[0])+genFunc.Diff2Nolm(input_Data[0],lspYsideK[0])
      #Lノード分やるよー
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K): #Lノード回の中からそれっぽいのを決める
        tmpY = genFunc.Diff2Nolm(data_u1,latent_spY[index_k])
        tmpSide = genFunc.Diff2Nolm(input_Data[indexN],lspYsideK[index_k])
        tmp = tmpY+ tmpSide
        if dist>tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn
    return(winner_Kn)

  # 勝者ノード(第２ノード)の選定_属性情報を含む(競合過程)
  # 引数 学習データ(L*Data数)、潜在空間、属性情報(N*D)、属性情報の潜在空間(K*D)
  # 戻り値 勝者ノード：array[データ次元数]
  def WinnerNodeL_sideInfo(self,latent_spU2,latent_spY,input_Data,lspYsideL):
    winner_Ln = np.zeros(len(latent_spU2[0]))
    latent_spU2T=latent_spU2.T
    latent_spYT=latent_spY.T
    for indexN,data_n in enumerate(latent_spU2T):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm(data_n,latent_spYT[0])+genFunc.Diff2Nolm(input_Data[0],lspYsideL[0])
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L): #インデックスと値の両方取れる
        tmpY = genFunc.Diff2Nolm(data_n,latent_spYT[index_l])
        tmpSide = genFunc.Diff2Nolm(input_Data[indexN],lspYsideL[index_l])
        tmp =tmpY +tmpSide 
        if dist>tmp:
          dist=tmp
          kn=index_l
      winner_Ln[indexN]=kn
    return(winner_Ln)

  #TODO_TSOM２用に編集
  # 協調過程（ノードと学習データとの学習割合を求める）
  # 引数:勝者ノード、潜在空間のノード値、潜在空間の座標(ノード数)、カウント
  # 戻り値　学習割合(全ノード X 入力データ数(勝者数))
  def CoordinProcess(self,winner_n,Node,node_sp,count):
    # 初期化
    Ykn_Ret = np.zeros((Node, len(winner_n))) #学習率(潜在空間 X データ数)
    sigma =genFunc.SigmaCalac(count)
    
    for index in range(Node):
        for index_N,win_n in enumerate(winner_n): #winner_Kn：勝者 
          #参照ベクトル[ノードK,ノードN]＝ガウス（潜在空間ノードkの位置：潜在空間(勝者の位置)、シグマ）
          Ykn_Ret[index,index_N]= genFunc.Gauss(node_sp[index],node_sp[int(win_n)],sigma)
          #課題ノードをインデックスで管理する　or 配列で整理する
    return Ykn_Ret

  #学習量の標準化
  # 引数 :学習割合標準化前R[Node＊win_N]
  # 引数 :学習割合R[Node＊win_N]
  def LearnStandardization(self,LearningValue):
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(LearningValue))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(LearningValue, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    
    #各ノードの学習割合の標準化
    for indexNode in range(len(LearningValue)):
      LearningValue[indexNode]=Yk_Rec[indexNode]*LearningValue[indexNode,:]
    return LearningValue

  # 適応過程
  # 引数 :学習割合R[Node＊win_N]、入力データ
  # 戻り値 :モデルの学習結果(潜在空間：ノードLorK X データ次元)
  def AdaptateProcessU1(self,CPretR_node_winN,in_data):
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(CPretR_node_winN))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(CPretR_node_winN, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    
    #各ノードの学習割合の標準化
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]
    return np.dot(in_data,CPretR_node_winN.T)

  # 適応過程
  # 引数 :学習割合R[Node＊win_N]、入力データ
  # 戻り値 :モデルの学習結果(潜在空間：ノードLorK X データ次元)
  def AdaptateProcessU2(self,CPretR_node_winN,in_data):
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(CPretR_node_winN))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(CPretR_node_winN, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    #各ノードの学習割合の標準化
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]
    return np.dot(CPretR_node_winN,in_data)
  
  # 適応過程
  # 引数 :学習割合R[Node＊win_N]、入力データ
  # 戻り値 :モデルの学習結果(潜在空間：ノードLorK X データ次元)
  def AdaptateProcessY(self,CPretR_node_winN,CPretR_nodeL_winN,in_data):
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(CPretR_node_winN))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(CPretR_node_winN, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    #各ノードの学習割合の標準化
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]
    return np.dot(CPretR_node_winN,in_data)
  