# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
#学習データData X K X L X 1 であると仮定する
import numpy as np
from . import generalFunction as genFunc
from . import tsom

class TSom2DirectType(tsom.TSom):
  """
  TSom2（直接型）の学習
  スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
  """

  def runTSom2(self, data, count):
    """
    TSom2（直接型）の学習の実行(３次元配列のTSom)
    スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
    @param data   学習データ(N * M * D) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]])
    @param count  学習数
    return　計算結果(K * L * D)
    """
    #ノード座標
    nodeK_coordinate_=self.NodeToCoordinate(self.NODE_KX, self.NODE_KY)
    nodeL_coordinate_=self.NodeToCoordinate(self.NODE_LX, self.NODE_LY)
    
    data_D = len(data.T)
    # 潜在空間Yの初期化 [K*L*D]
    #潜在空間の参照ベクトル初期化(TODO：PCA初期化でもいいよ)
    latent_spY_ = np.random.rand(self.NODE_K, self.NODE_L, data_D)

    #学習の実施
    count_ =0
    while count_< count:
      # 勝者決定
      self.win_nodeK = self.WinnerNodeK(data,latent_spY_)
      self.win_nodeL = self.WinnerNodeL(data,latent_spY_)

      # 協調過程
      learn_rate_K =self.CoordinationProcess(self.win_nodeK,self.NODE_K,nodeK_coordinate_,count_)
      learn_rate_L =self.CoordinationProcess(self.win_nodeL,self.NODE_L,nodeL_coordinate_,count_)

      learn_rate_K=self.LearnStandardization(learn_rate_K)
      learn_rate_L=self.LearnStandardization(learn_rate_L)
      
      # 適応過程
      # 潜在空間の更新
      latent_spY_ = self.AdaptateProcessY(learn_rate_K, learn_rate_L, data)

      count_+=1
      print(count_)
    #ループ処理おわり

    return latent_spY_

  def WinnerNodeK(self, data, latent_spY):
    """
    勝者ノード(第１ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param data         学習データ(N*M*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @return 勝者ノード    array[第１次元のデータ数(N)]
    """
    winner_Kn = np.zeros(len(data))
    #データ数分勝者の算出を繰り返す
    for indexN, data_n in enumerate(data):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Norm3D(data_n,latent_spY[0])
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K):
        tmp = genFunc.Diff2Norm3D(data_n,latent_spY[index_k])
        if dist>=tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn

    return(winner_Kn)

  def WinnerNodeL(self,data,latent_spY):
    """
    勝者ノード(第２ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param data         学習データ(N*M*D)  U2(K*M*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @return 勝者ノード    array[第2次元のデータ数(M)]
    """
    winner_Lm = np.zeros(len(data[0]))
    for index in range(len(data[0])):
      #初期値として各ノードの最初の値の差分を設定する
      lm = 0
      dist = genFunc.Diff2Norm3D(data[:,index],latent_spY[:,0])
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L):
        tmp = genFunc.Diff2Norm3D(data[:,index],latent_spY[:,index_l])
        if dist>tmp:
          dist = tmp
          lm = index_l
      winner_Lm[index] = lm
    return(winner_Lm)
  
  def AdaptateProcessY(self,CPretR_nodeK_winN,CPretR_nodeL_winM,in_data):
    """
    適応過程Y
    @param CPretR_nodeK_winN 標準化学習割合(第1ノード数(K) * 学習データの第1次元のデータ数(N))
    @param CPretR_nodeL_winM 標準化学習割合(第2ノード数(L) * 学習データの第2次元のデータ数(M))
    @param in_data          学習データ(N * M * D)
    @return モデルYの学習結果(潜在空間の第1ノード数(K) * 潜在空間の第2ノード数(L) * データ次元(D))
    """

    # 潜在空間の更新
    # データ次元ごとにノードLの更新値([L*M]*[M*N*1]^t->[L*N])を求め,Yに一時格納(D*[L*N])する。
    # ノードKの更新値([D*L*N]*[N*K]->[D*L*K])を求める。
    # 転置する事で変換([K*M*D])する
    Y_dln = np.zeros((len(in_data.T), self.NODE_L, len(in_data)))
    for index in range(len(in_data.T)):
      Y_dln[index]=(np.dot(CPretR_nodeL_winM, in_data[:,:,index].T))
    
    ret =np.dot(Y_dln, CPretR_nodeK_winN.T)
    return ret.T
