# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# 一次元のSOMだよ。プロト版として実装、次元数とか調整できたらいいよね
import numpy as np

#ここは固定値とする
SIGUM_MAX_ = 10
SIGUM_MIN_ = 0.5
Tau_ = 100

  #ガウス関数（データの次元数）
  #引数 (座標１,座標２,近傍半径)
  #戻り値　float
def Gauss(position1,position2,sigma):
  return np.exp(-Diff2Norm(position1,position2)/(2*sigma**2))

  #シグマの計算式(非線形に計算する)　
  #引数　(ステップ数)
  #戻り値　float
def SigmaCalac(time):
  sigma_= SIGUM_MAX_ * np.exp(-time/Tau_)
  return sigma_ if SIGUM_MIN_ < sigma_ else SIGUM_MIN_

# ２乗誤差を算出するメソッド(次元数：２次元)
# 引数：データ１、データ２
# 戻り値　２乗距離
def Diff2Norm(position1,position2):
  tmp=0
  for el in range(len(position1)):
    #print(el)
    tmp += np.square(position1[el]-position2[el])
  return np.sqrt(tmp)

def Diff2Norm3D(position1,position2):
  """
  勝者ノード(第１ノード)の選定(競合過程)を行う。勝者ノードとはデータから選出されたノードのことである。
  @param position1  データ１[M*D]
  @param position2  データ２[M*D]
  @return ２乗距離（スカラー）
  """
  tmp=0
  for el1 in range(len(position1)):
    for el2 in range(len(position1[el1])):
      tmp += np.square(position1[el1][el2]-position2[el1][el2])
  return np.sqrt(tmp) 