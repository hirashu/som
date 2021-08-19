# -*- coding: utf-8 -*-
# 上は日本語を記載するために必要なおまじない
# ３次元配列対応のTSom
import numpy as np
import lib
import json

#変換対象のパスの設定
PATH_DATA_CHARACTER_INFO = './tranceData/characterInfo.json'
PATH_DATA_ATTACK_TEAM_MEMBER_LIST = './tranceData/チーム構成_攻撃.json'
PATH_DATA_DEFENCE_TEAM_MEMBER_LIST = './tranceData/チーム構成_防御.json'

#変換後の保存パス
PATH_CREATE_DATA_ATTACK_TEAM_INFO_LIST = './tranceData/チーム構成情報_攻撃.json'
PATH_CREATE_DATA_DEFENCE_TEAM_INFO_LIST = './tranceData/チーム構成情報_防御.json'

class createLearningData():
  """
  作成するデータ一覧
  ・チーム構成キャラクター情報_攻撃.json(防御も)
  """
  
  def __init__(self):
    return
  
  def createTeamData(self):
    """
    チーム構成キャラクター情報の作成    
    チーム構成情報_攻撃.json、チーム構成情報_防御.jsonを作成する。
    """

    # データの読み込み
    characterInfo_open = open(PATH_DATA_CHARACTER_INFO, 'r')
    characterInfoList = json.load(characterInfo_open)
    characterInfo_open.close()

    attackTeamMember_open = open(PATH_DATA_ATTACK_TEAM_MEMBER_LIST, 'r')
    attackTeamMemberList = json.load(attackTeamMember_open)
    attackTeamMember_open.close()

    defenceTeamMember_open = open(PATH_DATA_DEFENCE_TEAM_MEMBER_LIST, 'r')
    defenceTeamMemberList = json.load(defenceTeamMember_open)
    defenceTeamMember_open.close()

    #チームリストの作成
    attackTeamList = self.createTeamFeaturesList(attackTeamMemberList,characterInfoList)
    defenceTeamList = self.createTeamFeaturesList(defenceTeamMemberList,characterInfoList)

    #NumpyEncoderを使用するためarrayをndarrayに変換する
    attackTeamListND = np.array(attackTeamList)
    defenceTeamListND = np.array(defenceTeamList)

    #データの書き込み
    file_open = open(PATH_CREATE_DATA_ATTACK_TEAM_INFO_LIST, 'w')
    dump = json.dumps(attackTeamListND, cls = lib.NumpyEncoder.NumpyEncoder)
    file_open.writelines(dump)
    file_open.close()

    file_open = open(PATH_CREATE_DATA_DEFENCE_TEAM_INFO_LIST, 'w')
    dump = json.dumps(defenceTeamListND, cls = lib.NumpyEncoder.NumpyEncoder)
    file_open.writelines(dump)
    file_open.close()


  def createTeamFeaturesList(self,teamMemberList, characterInfoList):
    """
    チーム特徴リストの作成
    @param teamMemberList    チームのメンバーリスト。チーム毎の編成情報をIDで管理しているリスト
    @param characterInfoList キャラクター情報のリスト
    @return チーム特徴リスト
    """

    teamList = []
    for teamMemberIds in teamMemberList:
      teamMemberList = []
      for memberId in teamMemberIds:
        characterFeatures = self.GetCharacterFeatures(characterInfoList, memberId)
        teamMemberList.append(characterFeatures)
      teamList.append(teamMemberList)
    return teamList

  def GetCharacterFeatures(self,characterInfoList, targetId):
    """
    キャラクター特徴の取得
    @param characterInfoList キャラクター情報のリスト
    @param targetId      　　 対象のキャラクターID
    @return 対象キャラクターの特徴情報
    """
    for characterInfo in characterInfoList:
      if characterInfo['id'] == targetId:
        return characterInfo['features']
