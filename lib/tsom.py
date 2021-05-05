# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
#今回の学習データData X K X L X 1 であると仮定する
import numpy as np
from . import generalFunction as genFunc
class TSom():

  def __init__(self,node_KX,node_KY,node_LX,node_LY):
    """
    初期化メソッド_潜在空間の設定
    @param node_KX   ノードKのX軸に設定する値
    @param node_KY   ノードKのY軸に設定する値
    @param node_LX   ノードLのX軸に設定する値
    @param node_LY   ノードLのY軸に設定する値
    """
    self.NODE_KX = node_KX
    self.NODE_KY = node_KY
    self.NODE_K = self.NODE_KX * self.NODE_KY
    self.NODE_LX = node_LX
    self.NODE_LY = node_LY
    self.NODE_L = self.NODE_LX * self.NODE_LY

  def runTSOM2(self, data,count):
    """
    TSOM2の学習の実行(３次元配列のTSOM)
    @param data   学習データ(N * M * D)
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]])
    @param count  学習数
    return　計算結果(K * L * D)
    """
    #ノード座標
    nodeK_coordinate_=self.NodeToCoordinate(self.NODE_KX,self.NODE_KY)
    nodeL_coordinate_=self.NodeToCoordinate(self.NODE_LX,self.NODE_LY)
    #潜在空間の参照ベクトル初期化(TODO：PCA初期化でもいいよ)
    data_D = len(data.T)
    # 潜在空間U1の初期化 [N*L*D]
    latent_spU1_ = np.random.rand(len(data),self.NODE_L,data_D)
    # 潜在空間U2の初期化 [K*M*D]
    latent_spU2_ = np.random.rand(self.NODE_K,len(data[0]),data_D)
    # 潜在空間Yの初期化 [K*L*D]
    latent_spY_ = np.random.rand(self.NODE_K,self.NODE_L,data_D)

    #学習の実施
    count_ =0
    while count_< count:
      print("latent_spU1_")
      print(latent_spU1_)
      print("latent_spY_")
      print(latent_spY_)
      # 勝者決定
      win_nodeK_ = self.WinnerNodeK(latent_spU1_,latent_spY_)
      win_nodeL_ = self.WinnerNodeL(latent_spU2_,latent_spY_)

      # 協調過程
      learn_rate_K =self.CoordinProcess(win_nodeK_,self.NODE_K,nodeK_coordinate_,count_)
      learn_rate_L =self.CoordinProcess(win_nodeL_,self.NODE_L,nodeL_coordinate_,count_)

      learn_rate_K=self.LearnStandardization(learn_rate_K)
      learn_rate_L=self.LearnStandardization(learn_rate_L)
      
      # 適応過程
      # 潜在空間の更新
      latent_spU1_ = self.AdaptateProcessU1(learn_rate_L,data)
      latent_spU2_=self.AdaptateProcessU2(learn_rate_K,data)
      latent_spY_= np.dot(latent_spU1_.T,learn_rate_K.T).T #[K*N]*[N*L*D]
      count_+=1
      print(count_)
    #ループ処理おわり

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

  def WinnerNodeK(self,latent_spU1,latent_spY):
    """
    勝者ノード(第１ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param latent_spU1  U1(N*L*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @return 勝者ノード    array[第１次元のデータ数(N)]
    """
    winner_Kn = np.zeros(len(latent_spU1))
    #データ数分勝者の算出を繰り返す
    for indexN,data_u1 in enumerate(latent_spU1):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm3D(data_u1,latent_spY[0])
      #Lノード分やるよー
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K): #Lノード回の中からそれっぽいのを決める
        tmp = genFunc.Diff2Nolm3D(data_u1,latent_spY[index_k])
        print("tmp")
        print(tmp)
        if dist>=tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn

    return(winner_Kn)

  def WinnerNodeL(self,latent_spU2,latent_spY):
    """
    勝者ノード(第２ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param latent_spU2  U2(K*M*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @return 勝者ノード    array[第2次元のデータ数(M)]
    """
    #ここの処理が正しいか確認する
    winner_Lm = np.zeros(len(latent_spU2[0]))
    #何と何の二乗誤差を求めたい？？
    for index in range(len(latent_spU2[0])):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[0])
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L): #インデックスと値の両方取れる
        tmp = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[index_l])
        if dist>tmp:
          dist=tmp
          kn=index_l
      winner_Lm[index]=kn
    return(winner_Lm)

  def CoordinProcess(self,winner_n,Node,node_sp,count):
    """
    協調過程（ノードと学習データとの学習割合を求める）
    @param winner_n   array[データ次元数]
    @param Node       潜在空間のノード 第１or第２ノード
    @param node_sp    第１or第２ノード座標
    @param count      学習数
    @return           学習割合(潜在空間のノード数 * 入力データ数)
    """
    # 初期化
    Ykn_Ret = np.zeros((Node, len(winner_n))) #学習率(潜在空間 X データ数)
    sigma =genFunc.SigmaCalac(count)
    
    for index in range(Node):
        for index_N,win_n in enumerate(winner_n): #winner_Kn：勝者 
          #参照ベクトル[ノードK,ノードN]＝ガウス（潜在空間ノードkの位置：潜在空間(勝者の位置)、シグマ）
          Ykn_Ret[index,index_N]= genFunc.Gauss(node_sp[index],node_sp[int(win_n)],sigma)

    return Ykn_Ret

  def LearnStandardization(self,LearningValue):
    """
    学習量の標準化_ノードの軸に問わられない
    @param LearningValue 学習割合_標準化前(潜在空間のノード数 * 入力データ数)
    @return 標準化学習量(潜在空間のノード数 * 入力データ数)
    """

    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec=np.sum(LearningValue, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    
    #各ノードの学習割合の標準化
    for indexNode in range(len(LearningValue)):
      LearningValue[indexNode]=Yk_Rec[indexNode]*LearningValue[indexNode,:]
    return LearningValue

  def AdaptateProcessU1(self,CPretR_node_winN,in_data):
    """ todo 以降組み直し
    適応過程U1
    @param CPretR_node_winN 標準化学習割合(第2ノード数(L) * 学習データの第２次元のデータ数(M))
    @param in_data          学習データ(N *M * D)
    @return モデルU1の学習結果(学習データの第１次元(N) * 潜在空間の第2ノード数(L) * データ次元(D))
    """
    # 標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec=np.sum(CPretR_node_winN.T, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    # 各ノードの学習割合の標準化
    # CPretR_node_winN (第2ノード数(L) * 学習データの第２次元のデータ数(M))
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]

    # 潜在空間の更新
    # データ次元ごとに更新値([L*M]*[M*N*1]^t->[L*N])を求め格納(D*[L*N])する。最後に転置する事で変換([N*L*D])する
    U1 = np.zeros((len(in_data.T),self.NODE_L,len(in_data)))
    for index in range(len(in_data.T)):
      U1[index]=(np.dot(CPretR_node_winN,in_data[:,:,index].T))
    return U1.T

  def AdaptateProcessU2(self,CPretR_node_winN,in_data):
    """
    適応過程U2
    @param CPretR_node_winN 標準化学習割合(第1ノード数(K) * 学習データの第1次元のデータ数(N))
    @param in_data          学習データ(N *M * D)
    @return モデルU2の学習結果(潜在空間の第1ノード数(K) * 学習データの第2次元(M) * データ次元(D))
    """
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(CPretR_node_winN))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(CPretR_node_winN, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)

    #各ノードの学習割合の標準化
    # CPretR_node_winN (第1ノード数(K) * 学習データの第1次元のデータ数(N))
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]
    
    # 潜在空間の更新
    # データ次元ごとに更新値([D*M*N]*[N*K]->[D*M*K])を求める。
    # 転置する事で変換([K*M*D])する
    ret =np.dot(in_data.T,CPretR_node_winN.T)

    return ret.T
  
  def AdaptateProcessY(self,CPretR_node_winN,CPretR_nodeL_winN,in_data):
    """
    適応過程Y
    @param CPretR_node_winN 標準化学習割合(第1ノード数(K) * 学習データの第1次元のデータ数(N))
    @param CPretR_nodeL_winN 標準化学習割合(第2ノード数(L) * 学習データの第2次元のデータ数(M))
    @param in_data          学習データ(N * M * D)
    @return モデルYの学習結果(潜在空間の第1ノード数(K) * 潜在空間の第2ノード数(L) * データ次元(D))
    """    
    #標準化を行うため行毎の和を求めて、逆数を計算する
    Yk_Rec = np.zeros(len(CPretR_node_winN))#学習量の逆数(潜在空間)
    Yk_Rec=np.sum(CPretR_node_winN, axis=1)
    Yk_Rec=np.reciprocal(Yk_Rec)
    #各ノードの学習割合の標準化
    for indexNode in range(len(CPretR_node_winN)):
      CPretR_node_winN[indexNode]=Yk_Rec[indexNode]*CPretR_node_winN[indexNode,:]
    return np.dot(CPretR_node_winN,in_data)
  