# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
import json

PATH_DATA_CHARACTER_INFO = './data/キャラクター特徴.json'

class Pca():
    
    def runTransform(self,data):
        print(data)
        print("データ表示")
        #データを読み込む
        pca = PCA(0.75,True) #引数1,累積寄与率の閾値、引数２　白色化有無 
        pca.fit(data)
        # 寄与率の表示
        print(len(pca.explained_variance_ratio_))
        print(pca.explained_variance_ratio_)
        #主成分空間への写像
        return pca.transform(data)