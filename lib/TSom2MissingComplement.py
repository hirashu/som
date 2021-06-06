# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
#今回の学習データData X K X L X 1 であると仮定する
import numpy as np
from . import generalFunction as genFunc
from . import TSom2DirectType

"""
TODO
以下の3つを行い再度情報を扱えるようにする
0.不要な継承メソッドを削除する //完了
1.サイド情報用の潜在空間を作成 //OK
2. 勝者ノードの選出でサイド情報も扱う //OK
3.サイド情報の潜在空間を更新する. 
4.動作確認を行う。
"""
class TSom2MissingComplement(TSom2DirectType.TSom2DirectType):
  """
  TSom2（直接型）の欠損データの学習
  スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
  学習データに欠損がある場合の学習
  """

  def runTSom2(self, data, missingBinaryData, count):
    """
    TSom2（直接型）の欠損データの学習の実行(３次元配列のTSom)
    スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
    @param data   学習データ(N * M * D) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]],[[0,0,0],[3,3,3],[5,5,5]])
    @param missingBinaryData   欠損データのバイナリ情報(N * M) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]])
    @param count  学習数
    return　計算結果(K * L * D)
    """
    #ノード座標
    nodeK_coordinate_=self.NodeToCoordinate(self.NODE_KX, self.NODE_KY)
    nodeL_coordinate_=self.NodeToCoordinate(self.NODE_LX, self.NODE_LY)
    #データ次元の取得
    data_D = len(data.T)
    # 潜在空間Yの初期化 [K*L*D]
    #潜在空間の参照ベクトル初期化(TODO：PCA初期化でもいいよ)
    latent_spY_ = np.random.rand(self.NODE_K, self.NODE_L, data_D)
    #勝者の初期化
    self.win_nodeK = np.random.randint(self.NODE_K - 1, size =(len(data)))

    #学習の実施
    count_ =0
    while count_< count:
      # 勝者決定
      self.win_nodeL = self.WinnerNodeLDirectType(data, latent_spY_, self.win_nodeK)
      self.win_nodeK = self.WinnerNodeKDirectType(data, latent_spY_, self.win_nodeL)

      # 協調過程
      learn_rate_K =self.CoordinationProcess(self.win_nodeK, self.NODE_K, nodeK_coordinate_, count_)
      learn_rate_L =self.CoordinationProcess(self.win_nodeL, self.NODE_L, nodeL_coordinate_, count_)

      learnWeight = self.StandardDiagReciprocal(learn_rate_K, learn_rate_L, missingBinaryData)
      
      # 適応過程
      # 潜在空間の更新
      latent_spY_ = self.AdaptateProcessY(learn_rate_K, learn_rate_L, learnWeight, data, missingBinaryData)

      count_+=1
      print(count_)
    #ループ処理おわり

    return latent_spY_

  def runTSom2sideInfo(self, data, missingBinaryData, sideInfo, count):
    """
    テンプレ
    TSom2（直接型）の欠損データの学習(属性情報あり)の実行(３次元配列のTSom)
    スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
    @param data   学習データ(N * M * D) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]],[[0,0,0],[3,3,3],[5,5,5]])
    @param missingBinaryData   欠損データのバイナリ情報(N * M) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]])
    @param count  学習数
    return　計算結果(K * L * D)
    """
    #ノード座標
    nodeK_coordinate_=self.NodeToCoordinate(self.NODE_KX, self.NODE_KY)
    nodeL_coordinate_=self.NodeToCoordinate(self.NODE_LX, self.NODE_LY)
    #データ次元の取得
    data_D = len(data.T)
    # 潜在空間Yの初期化 [K*L*D]
    #潜在空間の参照ベクトル初期化(TODO：PCA初期化でもいいよ)
    latent_spY_ = np.random.rand(self.NODE_K, self.NODE_L, data_D)

    #学習の実施
    count_ =0
    while count_< count:
      # 勝者決定
      self.win_nodeK = self.WinnerNodeK(data, latent_spY_)
      self.win_nodeL = self.WinnerNodeL(data, latent_spY_)

      # 協調過程
      learn_rate_K =self.CoordinationProcess(self.win_nodeK, self.NODE_K, nodeK_coordinate_, count_)
      learn_rate_L =self.CoordinationProcess(self.win_nodeL, self.NODE_L, nodeL_coordinate_, count_)

      learnWeight = self.LearnStandardization(learn_rate_K, learn_rate_L, missingBinaryData)
      
      # 適応過程
      # 潜在空間の更新
      latent_spY_ = self.AdaptateProcessY(learn_rate_K, learn_rate_L, learnWeight, data, missingBinaryData)

      count_+=1
      print(count_)
    #ループ処理おわり

    return latent_spY_

  def StandardDiagReciprocal(self, learningValueNodeK, learningValueNodeL, missingBinaryData):
    """
    学習量の標準化された対角行列の逆数を求める。
    @param LearningValueK 学習割合_標準化前(潜在空間のノード(第１ノード数：K) * 入力データ数(N))
    @param LearningValueL 学習割合_標準化前(潜在空間のノード(第２ノード数：L) * 入力データ数(M))
    @param missingBinaryData 学習データ(欠損あり)のバイナリデータ(N * M * D(=1))
    @return 標準学習量の重み(潜在空間の第１ノード(K) * 潜在空間の第２ノード(L) * D(=1)の逆数)
    """

    # バイナリ情報を含む学習量の重みを求める
    # データ次元ごとにノードLの更新値([L*M]*[M*N*(1)] ->[L*N])を求め,Yに一時格納(D*[L*N])する。
    # ノードKの更新値([D*L*N]*[N*K]->[D*L*K])を求める。
    # 転置する事で変換([K*M*D])する
    Yl_std = np.zeros((len(missingBinaryData.T), len(learningValueNodeL), len(missingBinaryData)))
    for index in range(len(missingBinaryData.T)):
      Yl_std[index]=(np.dot(learningValueNodeL, missingBinaryData[:,:,index].T))
    weightY_Diag_t =np.dot(Yl_std, learningValueNodeK.T)
    Y_Diag = weightY_Diag_t.T
    weightY=np.reciprocal(Y_Diag)
    return weightY

  def AdaptateProcessY(self, CPretR_nodeK_winN, CPretR_nodeL_winM, learnWeight, learnData, missingBinaryData):
    """
    適応過程Y
    @param CPretR_nodeK_winN 標準化学習割合(第1ノード数(K) * 学習データの第1次元のデータ数(N))
    @param CPretR_nodeL_winM 標準化学習割合(第2ノード数(L) * 学習データの第2次元のデータ数(M))
    @param learnWeight       標準学習量の重み(K * L * D(1))
    @param learnData         学習データ(N * M * D)
    @param missingBinaryData 学習データ(欠損あり)のバイナリデータ(N * M * D(1))
    @return モデルYの学習結果(潜在空間の第1ノード数(K) * 潜在空間の第2ノード数(L) * データ次元(D))
    """
    #学習データとバイナリデータの要素をかけて、学習データの重みづけを行う
    BinaryLearnData = learnData * missingBinaryData
    # 潜在空間の更新
    # データ次元ごとにノードLの更新値([L*M]*[M*N*1] -> [L*N])を求め,Yに一時格納(D*[L*N])する。
    # ノードKの更新値([D*L*N]*[N*K]->[D*L*K])を求める。
    # 転置する事で変換([K*M*D])させて、標準学習量の重み((K * L * D)と要素積を求める
    Yl_std = np.zeros((len(BinaryLearnData.T), self.NODE_L, len(BinaryLearnData)))
    for index in range(len(BinaryLearnData.T)):
      Yl_std[index]=(np.dot(CPretR_nodeL_winM, BinaryLearnData[:,:,index].T))
    Y_normal =np.dot(Yl_std, CPretR_nodeK_winN.T)
    ret = learnWeight * Y_normal.T
    return ret
  