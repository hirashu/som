# -*- coding: utf-8 -*-
"""
プリコネのデータを扱う用の関数
想定するデータは以下の3種類のデータ
1.チームごとの勝敗データ(1:勝ち、0:負け)
[[[1],[0],[1]],
 [[1],[1],[1]],
 [[1],[0],[0]]]
2.チームのキャラクター編成データ 
[["1","2","3","4","5"]
 ["1","2","6","4","5"]
 ["1","2","8","3","5"]]
3.キャラクター詳細データ(キャラクター.Json)
{
  "1":[1,0,1,...],
  "2":[1,0,1,...],
  "3":[1,0,1,...]
}
"""

def Diff2NolmTeamInCharacter(position1,position2):
  """
  チーム構成情報の２乗ノルムを求める
  @param position1  データ1 (D_member_count *D_member_info)
  @param position2  データ2 D_member_count *D_member_info)
  @return チーム構成の２乗ノルム    Float
  """
  tmp=0
  for el1 in range(len(position1)):
    for el2 in range(len(position1[el1])):
      tmp += np.square(position1[el1][el2]-position2[el1][el2])
  return np.sqrt(tmp)