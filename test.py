import pandas as pd
import numpy as np
#
# feat = pd.read_csv('D:\OneDrive\post graduate\Code Learning\Python\study\\temp_HLBDA\iono_feature.csv', header=None)
# temp=feat.head(6)
# s=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]
# sindex=[]
# for i in range(len(s)):
#     if s[i]==1  :
#         sindex.append(i)
#
# result=temp.loc[:,sindex]
# pass
# A=10
# for i in range(A):
#     print(i,'\t')

# a=np.zeros([3,4])
# b=np.ones([2,4])
# c=np.r_[a,b]
# temp=abs(-3)
# temp1=-3**2
# print(temp)
# print(temp1)
# x=np.arange(1, 35 , 1)
# print(x.mean())

pass
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

import pandas as pd
pd.DataFrame(data)

#实现归一化
scaler = MinMaxScaler() #实例化
scaler = scaler.fit(data) #fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data) #通过接口导出结果
print(result)

result_ = scaler.fit_transform(data) #训练和导出结果一步达成
scaler.inverse_transform(result) #将归一化后的结果逆转

# #使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
# data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
#
# scaler = MinMaxScaler(feature_range=[5,10]) #依然实例化
# result = scaler.fit_transform(data) #fit_transform一步导出结果
# print(result)

#当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
#此时使用partial_fit作为训练接口
#scaler = scaler.partial_fit(data)
