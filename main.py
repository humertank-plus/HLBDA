'''
Hyper Learning Binary Dragonfly Algorithm source code demo version
DOI: https://doi.org/10.1016/j.knosys.2020.106553
'''

'''
导入相应的包
'''
import np as np
from numpy.matlib import repmat
import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

'''
读入数据
'''
# 读入待分类数据,赋值给feat
#feat=pd.read_csv('D:\OneDrive\post graduate\Datasets\sonar\sonar_data.csv',header=None)
#feat = pd.read_csv('D:\OneDrive\post graduate\Datasets\ionosphere\iono_feature.csv', header=None)
feat = pd.read_csv('breast_data_normal.csv', header=None)#该文件是经过标准化处理的
#feat=pd.read_csv(,header=None)
# 读入分类标签,赋值给label
#label = pd.read_csv('D:\OneDrive\post graduate\Datasets\sonar\sonar_label.csv', header=None)
#label = pd.read_csv('D:\OneDrive\post graduate\Datasets\ionosphere\iono_label.csv', header=None)
label = pd.read_csv('label_data.csv', header=None)
#label = pd.read_csv('.csv', header=None)

'''
设置参数
'''
it = 100  # 最大迭代次数
N = 10  # 蜻蜓的个数
D = feat.shape[1]  # 特征数及维数
k = 5  # KNN中k的大小
pl = 0.4  # 个体学习率
gl = 0.7  # 种群学习率

'''
HLBDA
'''

#算法开始
print('开始进行HLBDA优化：\n')

# 初始化蜻蜓的位置X的函数initial_population,阈值为0.5，返回值为0、1的numpy
def initial_population(Number, Dimension):
    temp = np.zeros((N, D))
    for r in range(N):
        for c in range(D):
            if random.random() > 0.5:
                temp[r, c] = 1
    return temp


# 适应度函数fitness_funtion,返回值为计算好的适应值cost
def fitness_function(feat, label, X):
    # 设置alpha、beta?
    alpha = 0.99
    beta = 0.01
    #求出原始特征数量maxFeat
    maxFeat = feat.shape[1]
    # 求出feat经过选择特征后得到的的result
    xIndex = []
    for i in range(len(X)):
        if X[i] == 1:
            xIndex.append(i)

    result = feat.loc[:, xIndex]
    # 求错误率，使用KNN算法进行十倍交叉验证
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(result, label.values.ravel())
    acc = cross_val_score(knn, result, label.values.ravel(), cv=10)
    #    print('acc:',acc)
    err = 1 - np.mean(acc)
    #    print('err:',err)
    #选择的特征数量为Nsf个
    Nsf = sum(X == 1)
    #代价函数
    cost = alpha * err + beta * (Nsf / maxFeat)
    return cost


# 得到初始的蜻蜓位置(蜻蜓数量,初始维度)
X = initial_population(N, D)
# print('初始位置的矩阵形状为:', X.shape)

# 位移变量的初始化(蜻蜓数量,初始维度)
DX = np.zeros((N, D))

# 适应度值的初始化(存储每个维度的适应度值),D个变量的一维数组
fit = np.zeros((1, D))

# 食物源初始化
fitF = np.inf

# 捕食者源初始化
fitE = -np.inf

# 存储每次迭代的准确率
curve = np.zeros(it)

# 初始迭代次数
t = 1

# Dmax、Xnew意义不明
Xnew = np.zeros((N, D))
Dmax = 6

# 个体和群体
fitPB = np.ones((1, N))
fitPW = np.zeros((1, N))

# 初始化个体最优位置和最差位置(天敌)
Xpb = np.zeros((N, D))
Xpw = np.zeros((N, D))

# 开始迭代
while t <= 100:
    for i in range(N):#每只蜻蜓进行一次计算,接近食物源，远离天敌，更新Xpb、Xpw、Xf、Xe、fitPB、fitPW
        fit[0, i] = fitness_function(feat, label, X[i, :])
        if fit[0, i] < fitF:
            fitF = fit[0, i]
            Xf = X[i, :]
        if fit[0, i] > fitE:
            fitE = fit[0, i]
            Xe = X[i, :]
        if fit[0, i] > fitPW[0, i]:
            fitPW[0, i] = fit[0, i]
            Xpw[i, :] = X[i, :]
        if fit[0, i] < fitPB[0, i]:
            fitPB[0, i] = fit[0, i]
            Xpb[i, :] = X[i, :]
    # 赋值看不懂
    w = 0.9 - t * ((0.9 - 0.4) / it)
    rate = 0.1 - t * ((0.1 - 0) / (it / 2))
    s = 2 * random.random() * rate
    a = 2 * random.random() * rate
    c = 2 * random.random() * rate
    f = 2 * random.random()
    e = rate
    for i in range(N):#根据公式计算每只蜻蜓的位置
        index = 0
        nNeighbor = 1
        Xn = np.zeros((1, D))
        DXn = np.zeros((1, D))
        for j in range(N):
            if i != j:
                DXn = np.r_[DXn, [DX[j, :]]]
                #                DXn.r_[DXn,[DX[j,:]]]
                Xn = np.r_[Xn, [X[j, :]]]
                #                Xn.r_[Xn,[X[j,:]]]
                index = index + 1
                nNeighbor = nNeighbor + 1
        S = repmat(X[i, :], nNeighbor, 1) - Xn
        S = -sum(S, 1)
        A = sum(DXn, 1) / nNeighbor
        C = sum(Xn, 1) / nNeighbor
        C = C - X[i, :]
        F = ((Xpb[i, :] - X[i, :]) + (Xf - X[i, :])) / 2
        E = ((Xpw[i, :] + X[i, :]) + (Xe + X[i, :])) / 2
        for d in range(D):
            dX = (s * S[d] + a * A[d] + c * C[d] + f * F[d] + e * E[d]) + w * DX[i, d]
            #            dX(dX > Dmax) = Dmax;
            if dX > Dmax:
                dX = Dmax
            if dX < -Dmax:
                dX = -Dmax
            #            dX(dX < -Dmax) = -Dmax;
            DX[i, d] = dX
            TF = abs(DX[i, d] / math.sqrt((((DX[i, d]) ** 2) + 1)))
            r1 = random.random()
            if r1 >= 0 and r1 < pl:
                if random.random() < TF:
                    Xnew[i, d] = 1 - X[i, d]
                else:
                    Xnew[i, d] = X[i, d]
            elif r1 >= pl and r1 < gl:
                Xnew[i, d] = Xpb[i, d]
            else:
                Xnew[i, d] = Xf[d]
    X = Xnew
    curve[t - 1] = fitF
    print('\nIteration ', t, 'Best (HLBDA)= ', curve[t - 1])
    t = t + 1
Pos = np.arange(1, 35, 1)
PosIndex = []
for i in range(len(Pos)):
    if Xf[i] == 1:
        PosIndex.append(i)
sFeat = feat.loc[:, PosIndex]
Nf = len(PosIndex)
acc=1 - curve[it - 1]
print('初始特征数量为：',D)
print('选择的特征数量为：', Nf)
print('准确率为：', acc*100,'%')
