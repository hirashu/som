# coding: utf-8
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

json_list = []

# CSV ファイルの読み込み
# with open('/Users/iwatake/myproject/Pthon/プリコネ 色々 - キャラ性能_標準化test.csv', 'r') as f:
#     for row in csv.DictReader(f):
#         json_list.append(row)
        
# JSON ファイルへの書き込み
# with open('/Users/iwatake/myproject/Pthon/output.json', 'w') as f:
#     json.dump(json_list, f,ensure_ascii=False)

# JSONファイルのロード
with open('/Users/iwatake/myproject/Pthon/output_en.json', 'r') as f:
    json_output = json.load(f)
df = pd.DataFrame(json_output)
print(df)
dt = pd.DataFrame({'コク':[4,2,5,5], 'キレ':[5,0.4,4,3],'香り':[2,1,3,4]})
W,v=np.linalg.eig(df.corr())
pd.DataFrame(W)
pd.DataFrame(v)

#plt.scatter(v[:,2],v[:,1],c='black',s=200,alpha=0.3)

# print(df.columns)
for (i,j,k) in zip(v[:,2],v[:,3],df.columns):
        plt.plot(i,j,'o')
        plt.annotate(k, xy=(i, j))

plt.xlabel("PC2")
plt.ylabel("PC3")
plt.show()
rdf = np.outer(W,v)
print(df.corr())
# header = next(f)