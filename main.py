# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
import numpy as np
import lib
import json

# 定数定義
#ここは変数で指定できるようにする
NODE_X = 5
NODE_Y = 1
NODE_K = NODE_X * NODE_Y

# 関数化（入力データ）
#data形式はこれ（入力データは外部からインプットする）
data=np.array([[-5,-5,-5],[-3,-3,-3],[0,0,0],[3,3,3],[10,10,10]])

#学習データは　X:-1~1,Y:-1~1のものとする
data_kura=np.array([
  #[-1,-0.5,0,0.5,1]
  [-1,0.25,0,0.25,1], #-1
  [0.25,-0.5,-0.75,-0.5,0.25], #-0.5
  [0,-0.75,-1,-0.75,0], #-0
  [0.25,-0.5,-0.75,-0.5,0.25], #0.5
  [1,0.25,0,0.25,1]  #1
])

data_kuras=np.array([
  #[-1,-0.5,0,0.5,1]
  [-1,-0.75,-0.5,-0.25,0], #-1
  [-0.75,-0.5,-0.25,0,0.25], #-0.5
  [-0.5,-0.25,0,0.25,0.5], #-0
  [-0.25,0,0.25,0.5,0.75], #0.5
  [0,0.25,0.5,0.75,1]  #1
])

data_kuratest=np.array([
  [0,0.25,0.5,0.7,1]
])
data_kuratest2=np.array([
  [0],
  [0.25],
  [0.5],
  [0.7],
  [1]
])



#データの読み込み

#json_open = open('/Users/iwatake/myproject/Pthon/プリコネ 勝敗.json', 'r')#todo 相対座標化しておく
#json_load = json.load(json_open)
#print(json_load)

#print("学習データ\n" + str(data_kura))
#print(data)
#ret=TSOM_Unit.TSOM2(json_load,10,10,10,10)
#print("学習結果\n"+str(ret))
#print('学習結果\n'+str(som_unit.SOM(data,NODE_X,NODE_Y)))

som = lib.som.Som(NODE_X, NODE_Y)
ret = som.runSOM(data, 300)
print("学習結果\n"+str(ret))