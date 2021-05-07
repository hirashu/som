# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
#今回の学習データData X K X L X 1 であると仮定する
import numpy as np
from . import generalFunction as genFunc
from . import tsom

"""
TODO
以下の3つを行い再度情報を扱えるようにする
0.不要な継承メソッドを削除する //完了
1.サイド情報用の潜在空間を作成 //OK
2. 勝者ノードの選出でサイド情報も扱う //OK
3.サイド情報の潜在空間を更新する
"""
class TSomSideInfo(tsom.TSom):
  """
  属性情報を扱うTSOM
  # メモ
  # データ構成
  # メインマップ：キャラクター編成ごとの勝率マップ [Aチーム * Bチーム * 勝敗(1or0)] データ次元が1次元でも表せるため、ある意味2次元行列
  # data=np.array([[1,0,0],[0,1,0],[0,0,0]]) ->　３次元で表せばいいので問題ないはず。
  # サイドマップ：チーム特徴マップ 編成キャラクターによるチームの特徴マップ
  # データ：[チーム * 編成キャラ *　キャラクターパラメータ ]
  """

  def runTSOM2SideInfo(self, data, sideDataNodeK, sideDataNodeL, count):
    """
    プリコネ用入力データ(再度情報)のデータ構造を設定
    TSOM2の学習の実行(３次元配列のTSOM)
    @param data   学習データ(N * M * D)
    サンプル data=np.array([[[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]],
                           [[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]]])
    @param sideDataNodeK  サイド情報_ノードK側(N * D_sid)
    @param sideDataNodeL  サイド情報_ノードL側(M * D_sid)
    サンプル data=np.array([[[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]],
                          [-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]]]) 
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

    # 属性情報の潜在空間(ノードK)の初期化 [K*Dm*D]
    lspYsideK_ = np.random.rand(self.NODE_K,len(sideDataNodeK[0]),len(sideDataNodeK[0,0]))
    # 属性情報の潜在空間(ノードL)の初期化 [L*Dm*D]
    lspYsideL_ = np.random.rand(self.NODE_L,len(sideDataNodeL[0]),len(sideDataNodeL[0,0]))

    #学習の実施
    count_ =0
    while count_< count:
      # 勝者決定
      self.win_nodeK = self.WinnerNodeK_sideInfo(latent_spU1_,latent_spY_,sideDataNodeK,lspYsideK_)
      self.win_nodeL = self.WinnerNodeL_sideInfo(latent_spU2_,latent_spY_,sideDataNodeL,lspYsideL_)

      # 協調過程
      learn_rate_K =self.CoordinProcess(self.win_nodeK,self.NODE_K,nodeK_coordinate_,count_)
      learn_rate_L =self.CoordinProcess(self.win_nodeL,self.NODE_L,nodeL_coordinate_,count_)

      learn_rate_K=self.LearnStandardization(learn_rate_K)
      learn_rate_L=self.LearnStandardization(learn_rate_L)
      
      # 適応過程
      # 潜在空間の更新
      latent_spU1_ = self.AdaptateProcessU1(learn_rate_L,data)
      latent_spU2_=self.AdaptateProcessU2(learn_rate_K,data)
      latent_spY_= np.dot(latent_spU1_.T,learn_rate_K.T).T #[K*N]*[N*L*D]
      #潜在空間（属性情報）の更新 TODO 追加項目
      lspYsideK_= np.dot(learn_rate_K,sideData)
      lspYsideL_= np.dot(learn_rate_L,sideData)
      count_+=1
      print(count_)
    #ループ処理おわり

    return latent_spY_

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
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K): #Lノード回の中からそれっぽいのを決める
        tmp = genFunc.Diff2Nolm3D(data_u1,latent_spY[index_k])
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
    winner_Lm = np.zeros(len(latent_spU2[0]))
    for index in range(len(latent_spU2[0])):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[:,0])
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L): #インデックスと値の両方取れる
        tmp = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[:,index_l])
        if dist>tmp:
          dist=tmp
          kn=index_l
      winner_Lm[index]=kn

    return(winner_Lm)

  def WinnerNodeK_sideInfo(self,latent_spU1,latent_spY,sideDataNodeK,lspYsideK):
    """
    サイド情報を用いて勝者ノード(第１ノード)の選定(競合過程)を行う。
    @param latent_spU1    U1(N*L*D)
    @param latent_spY     潜在空間Y(K*L*D)
    @param sideDataNodeK  属性情報(N*D:正確には(N * D_member_count *D_member_info))
    @param lspYsideK      属性情報の潜在空間(K*D:正確には(K * D_member_count *D_member_info))
    @return 勝者ノード      array[第１次元のデータ数(N)]
    """
    winner_Kn = np.zeros(len(latent_spU1))
    #データ数分勝者の算出を繰り返す
    for indexN,data_u1 in enumerate(latent_spU1):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm3D(data_u1,latent_spY[0])
      mpSide = genFunc.Diff2Nolm3D(sideDataNodeK[0],lspYsideK[0])
      dist = dist+ tmpSide
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K):
        tmpY = genFunc.Diff2Nolm3D(data_u1,latent_spY[index_k])
        tmpSide = genFunc.Diff2Nolm3D(sideDataNodeK[index_k],lspYsideK[index_k])
        tmp = tmpY+ tmpSide
        if dist>=tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn

    return(winner_Kn)

  def WinnerNodeL_sideInfo(self,latent_spU2,latent_spY,sideDataNodeL,lspYsideL):
    """
    サイド情報を用いて勝者ノード(第２ノード)の選定(競合過程)を行う。
    @param latent_spU2  U2(K*M*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @param sideDataNodeL  属性情報(M*D:正確には(M * D_member_count *D_member_info))
    @param lspYsideL      属性情報の潜在空間(L*D:正確には(K * D_member_count *D_member_info))
    @return 勝者ノード    array[第2次元のデータ数(M)]
    """
    #ここの処理が正しいか確認する
    winner_Lm = np.zeros(len(latent_spU2[0]))
    #何と何の二乗誤差を求めたい？？
    for index in range(len(latent_spU2[0])):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[:,0])
      mpSide = genFunc.Diff2Nolm3D(sideDataNodeL[0],lspYsideL[0])
      dist = dist+ tmpSide
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L): #インデックスと値の両方取れる
        tmpY = genFunc.Diff2Nolm3D(latent_spU2[:,index],latent_spY[:,index_l])
        tmpSide = genFunc.Diff2Nolm3D(sideDataNodeL[index_k],lspYsideL[index_k])
        tmp = tmpY+ tmpSide
        if dist>tmp:
          dist=tmp
          kn=index_l
      winner_Lm[index]=kn

    return(winner_Lm)
