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

  def runTSom2sideInfo(self, data, missingBinaryData, sideDataNodeK, sideDataNodeL, count):
    """
    TSom2（直接型）の欠損データの学習の実行(３次元配列のTSom)
    スライス多様体(U)を作成せず、インスタンス多様体(Y)のみで学習を行う。
    @param data   学習データ(N * M * D) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]],[[0,0,0],[3,3,3],[5,5,5]])
    @param missingBinaryData   欠損データのバイナリ情報(N * M) [Arrayデータ] リストは無理
    サンプル data=np.array([[-5,-5,-5],[-3,-3,-3]])
    @param sideDataNodeK  サイド情報_ノードK側(N * D_sid(5 * D_memberInfo))
    @param sideDataNodeL  サイド情報_ノードL側(M * D_sid(5 * D_memberInfo))
    サンプル sideData=np.array([[-5,-5,-5],　[-3,-3,-3], [0,0,0]]) 
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

    # 属性情報の潜在空間(ノードK)の初期化 [K*D_sideKD_sid(5 * D_memberInfo)]
    lspYsideK_ = np.random.rand(self.NODE_K,len(sideDataNodeK[0]), len(sideDataNodeK[0][0]))
    # 属性情報の潜在空間(ノードL)の初期化 [L*D_sideLD_sid(5 * D_memberInfo)]
    lspYsideL_ = np.random.rand(self.NODE_L,len(sideDataNodeL[0]),len(sideDataNodeL[0][0]))

    #学習の実施
    count_ =0
    while count_< count:
      # 勝者決定
      self.win_nodeL = self.WinnerNodeLDirectTypeSideInfo(data, latent_spY_, sideDataNodeL, lspYsideL_, self.win_nodeK)
      self.win_nodeK = self.WinnerNodeKDirectTypeSideInfo(data, latent_spY_, sideDataNodeK, lspYsideK_, self.win_nodeL)

      # 協調過程
      learn_rate_K =self.CoordinationProcess(self.win_nodeK, self.NODE_K, nodeK_coordinate_, count_)
      learn_rate_L =self.CoordinationProcess(self.win_nodeL, self.NODE_L, nodeL_coordinate_, count_)

      learnWeight = self.StandardDiagReciprocal(learn_rate_K, learn_rate_L, missingBinaryData)
      
      # 適応過程
      # 潜在空間の更新
      latent_spY_ = self.AdaptateProcessY(learn_rate_K, learn_rate_L, learnWeight, data, missingBinaryData)

      #潜在空間（属性情報）の更新(学習量の標準化 -> 更新)
      learn_rate_K=self.LearnStandardization(learn_rate_K)
      learn_rate_L=self.LearnStandardization(learn_rate_L)
      self.lspYsideK_= self.AdaptateProcessSideInfo(learn_rate_K, sideDataNodeK)
      self.lspYsideL_= self.AdaptateProcessSideInfo(learn_rate_L, sideDataNodeL) 

      count_+=1
      print(count_)
    #ループ処理おわり

    return latent_spY_

  def WinnerNodeKDirectTypeSideInfo(self, data, latent_spY, sideDataNodeK, lspYsideK, winner_Lm):
    """
    勝者ノード(第１ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param  data         学習データ(N*M*D)
    @param  latent_spY   潜在空間Y(K*L*D)
    @param  sideDataNodeK   属性情報_ノードK側(N * D_sidD_sid(5 * D_memberInfo))
    @param  lspYsideK   属性情報の潜在空間(ノードK)の初期化 [K*D_sideKD_sid(5 * D_memberInfo)]
    @param  winner_Lm   array[データ次元数(M)]
    @return 勝者ノード    array[第１次元のデータ数(N)]
    """
    winner_Kn = np.zeros(len(data))
    #データ数分勝者の算出を繰り返す
    for indexN, data_n in enumerate(data):
      #初期値として各ノードの最初の値の差分を設定する
      kn=0
      dist = genFunc.Diff2Norm3DUseWinnerNode(data_n, latent_spY[0], winner_Lm)
      # todo サイド情報の重みを考える（現状だと次元数の問題でサイド情報が軽いはず。比率を測ればいいけど。） 
      distSide = genFunc.Diff2Norm3D(sideDataNodeK[indexN],lspYsideK[0])
      dist = dist + distSide
      #ノードとデータから最小の差となるノードを選択する
      for index_k in range(self.NODE_K):
        tmp = genFunc.Diff2Norm3DUseWinnerNode(data_n, latent_spY[index_k], winner_Lm)
        # todo サイド情報の重みを変更
        tmpSide = genFunc.Diff2Norm3D(sideDataNodeK[indexN],lspYsideK[index_k])
        tmp = tmp + tmpSide
        if dist>=tmp:
          dist=tmp
          kn=index_k
      winner_Kn[indexN]=kn

    return(winner_Kn)

  def WinnerNodeLDirectTypeSideInfo(self,data,latent_spY, sideDataNodeL, lspYsideL, winner_Kn):
    """
    勝者ノード(第２ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
    @param data         学習データ(N*M*D)  U2(K*M*D)
    @param latent_spY   潜在空間Y(K*L*D)
    @param  sideDataNodeL  属性情報_ノードL側(M * D_sidD_sid(5 * D_memberInfo))
    @param  lspYsideL   属性情報の潜在空間(ノードL)の初期化 [L*D_sideLD_sid(5 * D_memberInfo)]
    @param  winner_Kn   array[データ次元数(N)]
    @return 勝者ノード    array[第2次元のデータ数(M)]
    """
    winner_Lm = np.zeros(len(data[0]))
    for index in range(len(data[0])):
      #初期値として各ノードの最初の値の差分を設定する
      lm = 0
      dist = genFunc.Diff2Norm3DUseWinnerNode(data[:,index],latent_spY[:,0],winner_Kn)
      # todo サイド情報の重みを考える（現状だと次元数の問題でサイド情報が軽いはず。比率を測ればいいけど。） 
      distSide = genFunc.Diff2Norm3D(sideDataNodeL[index],lspYsideL[0])
      dist = dist + distSide
      #ノードとデータから最小の差となるノードを選択する
      for index_l in range(self.NODE_L):
        tmp = genFunc.Diff2Norm3DUseWinnerNode(data[:,index],latent_spY[:,index_l],winner_Kn)
        # todo サイド情報の重みを変更
        tmpSide = genFunc.Diff2Norm3D(sideDataNodeL[index],lspYsideL[index_l])
        tmp = tmp + tmpSide
        if dist>tmp:
          dist = tmp
          lm = index_l
      winner_Lm[index] = lm
    return(winner_Lm)
 

  def StandardDiagReciprocal(self, learningValueNodeK, learningValueNodeL, missingBinaryData):
    """ Todo サイド情報を扱うように修正する
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

  def AdaptateProcessSideInfo(self, learnRateNode, sideData):
    """
    適応過程サイド(属性)情報
    @param learnRateNode 標準化学習割合(第1ノード数(K) * 学習データの第1次元のデータ数(N or M))
    @param sideData      サイド情報(データ数(N or M) * メンバー数(5固定) * データ次元(D))
    @return 属性情報のインスタンスの学習結果(潜在空間のノード数(K or L) * メンバー数(5固定) * データ次元(D))
    """
    # [K(L)*M]*[N(M)*5*D]を以下の方法で行う
    # 両方を転置し、最後に転置を行う。
    # 1.([K(L)*M]_t*[N(M)*5*D]_t -> [D*5*N(M)] *[N(M) *K(L)])
    # D次元ごとに更新値([5*N(M)]*[N(M)*K(L)]->[5*K(L)])を求め、[D*5*K(L)]を求める。
    # 2.転置する事で変換([K(L)*5*D])させて求める。
    learnRateNodeT = learnRateNode.T
    Y_sideT = np.zeros((len(sideData.T), len(sideData[0]), len(learnRateNode)))
    for index, sideDataT_n in enumerate(sideData.T):
      Y_sideT[index]= np.dot(sideDataT_n, learnRateNodeT)
    return Y_sideT.T