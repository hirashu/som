# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
import numpy as np
import lib
import json

#データパスの設定
PATH_DATA_KURA = './data/kura.json'
PATH_DATA_TEAM_MATCH_RESULT = './data/チーム勝敗結果_3d.json'
PATH_DATA_TEAM_MATCH_RESULT_BINARY = './data/チーム勝敗結果バイナリ_3d.json'
PATH_DATA_TEAM_COMPOSITION_SIDE_K = './data/チーム構成キャラクター情報_攻撃.json'
PATH_DATA_TEAM_COMPOSITION_SIDE_L = './data/チーム構成キャラクター情報_防御.json'
#以下今のところ未使用
#PATH_DATA_TEAM_COMPOSTION = './data/チーム構成.json'

#学習結果の保存パス
PATH_RESULT_DATA_MAIN = './resultData/学習結果メインインスタンス.json'
PATH_RESULT_SIDE_NODE_K = './resultData/学習結果サイド情報K.json'
PATH_RESULT_SIDE_NODE_L = './resultData/学習結果サイド情報L.json'
PATH_RESULT_WINNER_NODE_K = './resultData/学習結果勝者K.json'
PATH_RESULT_WINNER_NODE_L = './resultData/学習結果勝者L.json'

#PCAの変換結果
PATH_DATA_CHARACTER_FEATURE = './data/キャラクター特徴.json'
PATH_RESULT_CHARACTER_FEATURE_PCA = './resultData/キャラクター特徴PCA.json'

# 定数定義
COUNT = 300
#SOM用
NODE_X = 10
NODE_Y = 1

#TSOM用
NODE_KX = 15 #エラーとなる
NODE_KY = 15
NODE_LX = 15
NODE_LY = 15

#モード切り替え
IS_CREATE_DATA = False
IS_PCA_RUN = True
IS_DATA_LEARNING =False

# 関数化（入力データ）
#data形式はこれ（入力データは外部からインプットする）
data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[5,5,5]])

#学習データは　X:-1~1,Y:-1~1のものとする
data_kura=np.array([
[[-1,-1,-1], [-0.75,-1,-0.5625], [-0.5,-1,-0.25], [-0.25,-1,-0.0625], [0,-1,0], [0.25,-1,-0.0625], [0.5,-1,-0.25], [0.75,-1,-0.5625], [1,-1,-1]],
[[-1,-0.75,-0.5625], [-0.75,-0.75,-0.125], [-0.5,-0.75,0.1875], [-0.25,-0.75,0.375], [0,-0.75,0.4375], [0.25,-0.75,0.375], [0.5,-0.75,0.1875], [0.75,-0.75,-0.125], [1,-0.75,-0.5625]],
[[-1,-0.5,-0.25], [-0.75,-0.5,0.1875], [-0.5,-0.5,0.5], [-0.25,-0.5,0.6875], [0,-0.5,0.75], [0.25,-0.5,0.6875], [0.5,-0.5,0.5], [0.75,-0.5,0.1875], [1,-0.5,-0.25]],
[[-1,-0.25,-0.0625], [-0.75,-0.25,0.375], [-0.5,-0.25,0.6875], [-0.25,-0.25,0.875], [0,-0.25,0.9375], [0.25,-0.25,0.875], [0.5,-0.25,0.6875], [0.75,-0.25,0.375], [1,-0.25,-0.0625]],
[[-1,0,0], [-0.75,0,0.4375], [-0.5,0,0.75], [-0.25,0,0.9375], [0,0,1], [0.25,0,0.9375], [0.5,0,0.75], [0.75,0,0.4375], [1,0,0]],
[[-1,0.25,-0.0625], [-0.75,0.25,0.375], [-0.5,0.25,0.6875], [-0.25,0.25,0.875], [0,0.25,0.9375], [0.25,0.25,0.875], [0.5,0.25,0.6875], [0.75,0.25,0.375], [1,0.25,-0.0625]],
[[-1,0.5,-0.25], [-0.75,0.5,0.1875], [-0.5,0.5,0.5], [-0.25,0.5,0.6875], [0,0.5,0.75], [0.25,0.5,0.6875], [0.5,0.5,0.5], [0.75,0.5,0.1875], [1,0.5,-0.25]],
[[-1,0.75,-0.5625], [-0.75,0.75,-0.125], [-0.5,0.75,0.1875], [-0.25,0.75,0.375], [0,0.75,0.4375], [0.25,0.75,0.375], [0.5,0.75,0.1875], [0.75,0.75,-0.125], [1,0.75,-0.5625]],
[[-1,1,-1], [-0.75,1,-0.5625], [-0.5,1,-0.25], [-0.25,1,-0.0625], [0,1,0], [0.25,1,-0.0625], [0.5,1,-0.25], [0.75,1,-0.5625], [1,1,-1]]
])

data_kura_missing=np.array([
[[-1,-1,-1], [-0.75,-1,-0.5625], [0,0,0], [-0.25,-1,-0.0625], [0,-1,0], [0.25,-1,-0.0625], [0.5,-1,-0.25], [0.75,-1,-0.5625], [1,-1,-1]],
[[-1,-0.75,-0.5625], [-0.75,-0.75,-0.125], [-0.5,-0.75,0.1875], [-0.25,-0.75,0.375], [0,-0.75,0.4375], [0.25,-0.75,0.375], [0.5,-0.75,0.1875], [0.75,-0.75,-0.125], [1,-0.75,-0.5625]],
[[-1,-0.5,-0.25], [-0.75,-0.5,0.1875], [0,0,0], [-0.25,-0.5,0.6875], [0,-0.5,0.75], [0.25,-0.5,0.6875], [0.5,-0.5,0.5], [0.75,-0.5,0.1875], [1,-0.5,-0.25]],
[[-1,-0.25,-0.0625], [-0.75,-0.25,0.375], [-0.5,-0.25,0.6875], [-0.25,-0.25,0.875], [0,-0.25,0.9375], [0.25,-0.25,0.875], [0.5,-0.25,0.6875], [0.75,-0.25,0.375], [1,-0.25,-0.0625]],
[[-1,0,0], [-0.75,0,0.4375], [-0.5,0,0.75], [-0.25,0,0.9375], [0,0,1], [0.25,0,0.9375], [0.5,0,0.75], [0.75,0,0.4375], [1,0,0]],
[[-1,0.25,-0.0625], [-0.75,0.25,0.375], [-0.5,0.25,0.6875], [-0.25,0.25,0.875], [0,0.25,0.9375], [0.25,0.25,0.875], [0.5,0.25,0.6875], [0.75,0.25,0.375], [1,0.25,-0.0625]],
[[-1,0.5,-0.25], [-0.75,0.5,0.1875], [-0.5,0.5,0.5], [-0.25,0.5,0.6875], [0,0.5,0.75], [0.25,0.5,0.6875], [0.5,0.5,0.5], [0.75,0.5,0.1875], [1,0.5,-0.25]],
[[-1,0.75,-0.5625], [0,0,0], [-0.5,0.75,0.1875], [-0.25,0.75,0.375], [0,0.75,0.4375], [0.25,0.75,0.375], [0.5,0.75,0.1875], [0.75,0.75,-0.125], [1,0.75,-0.5625]],
[[-1,1,-1], [-0.75,1,-0.5625], [-0.5,1,-0.25], [-0.25,1,-0.0625], [0,0,0], [0.25,1,-0.0625], [0.5,1,-0.25], [0.75,1,-0.5625], [1,1,-1]]
])

data_kuraBinary=np.array([
[[1], [1], [0], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [1], [1], [1], [1], [1]],
[[1], [1], [0], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [1], [1], [1], [1], [1]],
[[1], [0], [1], [1], [1], [1], [1], [1], [1]],
[[1], [1], [1], [1], [0], [1], [1], [1], [1]]
])

dataA=np.array([[-5,-5,-5],[-3,-3,-3],
                [3,3,3],[5,5,5]])
dataB=np.array([[1,2],
                 [3,4]])


if IS_CREATE_DATA:
  #データの作成
  createLearningData = lib.createLearningData.createLearningData()
  createLearningData.createTeamData()
  exit

if IS_PCA_RUN:
  # データの読み込み
  characterInfo_open = open(PATH_DATA_CHARACTER_FEATURE, 'r')
  characterInfo = json.load(characterInfo_open)
  characterInfo_open.close()
  
  #PCAの実行
  pca = lib.pca.Pca()
  resultPCA = pca.runTransform(characterInfo)

  #データ書き込み
  file_open = open(PATH_RESULT_CHARACTER_FEATURE_PCA, 'w')
  dump = json.dumps(resultPCA, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()

if IS_DATA_LEARNING:
  # データの読み込み
  teamMatchResult_open = open(PATH_DATA_TEAM_MATCH_RESULT, 'r')
  teamMatchResult = json.load(teamMatchResult_open)
  teamMatchResult_open.close()
  print(teamMatchResult)

  teamMatchResultBinary_open = open(PATH_DATA_TEAM_MATCH_RESULT_BINARY, 'r')
  teamMatchResultBinary = json.load(teamMatchResultBinary_open)
  teamMatchResultBinary_open.close()
  print(teamMatchResultBinary)

  teamCompositionSideK_open = open(PATH_DATA_TEAM_COMPOSITION_SIDE_K, 'r')
  teamCompositionSideK = json.load(teamCompositionSideK_open)
  teamCompositionSideK_open.close()
  print(teamCompositionSideK)

  teamCompositionSideL_open = open(PATH_DATA_TEAM_COMPOSITION_SIDE_L, 'r')
  teamCompositionSideL = json.load(teamCompositionSideL_open)
  teamCompositionSideL_open.close()
  print(teamCompositionSideL)

  # 読み込んだリストをArrayに変換
  teamMatchResultArray = np.array(teamMatchResult)
  teamMatchResultBinaryArray = np.array(teamMatchResultBinary)
  teamCompositionSideKArray = np.array(teamCompositionSideK)
  teamCompositionSideLArray = np.array(teamCompositionSideL)

  tSomDirectMCSide = lib.TSom2MissingComplement.TSom2MissingComplement(NODE_KX, NODE_KY, NODE_LX, NODE_LY)
  retMC = tSomDirectMCSide.runTSom2sideInfo(teamMatchResultArray, teamMatchResultBinaryArray,teamCompositionSideKArray, teamCompositionSideLArray ,COUNT)
  print("学習結果_retMC\n"+str(retMC))
  print("勝者ノードK\n")
  print(tSomDirectMCSide.win_nodeK)
  print("勝者ノードL\n")
  print(tSomDirectMCSide.win_nodeL)

  #データ書き込み
  #メインインスタンス
  file_open = open(PATH_RESULT_DATA_MAIN, 'w')
  dump = json.dumps(retMC, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()

  #属性情報K
  file_open = open(PATH_RESULT_SIDE_NODE_K, 'w')
  dump = json.dumps(tSomDirectMCSide.lspYsideK_, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()

  #属性情報L
  file_open = open(PATH_RESULT_SIDE_NODE_L, 'w')
  dump = json.dumps(tSomDirectMCSide.lspYsideL_, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()

  #勝者ノードK
  file_open = open(PATH_RESULT_WINNER_NODE_K, 'w')
  dump = json.dumps(tSomDirectMCSide.win_nodeK, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()

  #勝者ノードL
  file_open = open(PATH_RESULT_WINNER_NODE_L, 'w')
  dump = json.dumps(tSomDirectMCSide.win_nodeL, cls = lib.NumpyEncoder.NumpyEncoder)
  file_open.writelines(dump)
  file_open.close()
  print("main終わり")
