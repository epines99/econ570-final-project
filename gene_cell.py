#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("deng-logcounts.csv")
df.head


# In[5]:


# 获取细胞的特征
cell_feature = df.columns[1:]
print(cell_feature)
print(len(cell_feature))


# In[6]:


df_ans = pd.read_csv("deng-celltype6.csv")
print(df_ans.head)
ans_labels = list(df_ans.loc[:,'x'])

for i in range(1,7):
    print("Type %d 的细胞种类数为 %d" % (i, ans_labels.count(i)))


# In[5]:


# (1) 无监督学习 - 聚类
# 使用sklearn中的kmeans算法
from sklearn.cluster import KMeans
X_train = df.loc[:,list(cell_feature)].to_numpy().T # 转换为numpy数组
# print(X_train.shape)
n_clusters = 6
kmeans = KMeans(n_clusters = n_clusters).fit(X_train)
kmeans_labels = kmeans.labels_
# print(len(kmeans.labels_))


# In[6]:


# (2) 无监督学习 - 聚类
# 使用手写实现的kmeans算法

def randCent(dataMat, k):
    # 获取样本数与特征值
    m, n = np.shape(dataMat)
    # 初始化质心,创建以零填充的k*n矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    # 返回质心
    return centroids

#kmeans 聚类的实现
def distEclud(array1,array2):
    return np.sqrt(np.sum(np.power(array1 - array2, 2)))

def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    # 获取样本数和特征数
    m, n = np.shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:第一列记录簇索引值,第二列存储误差(误差指当前点到簇质心的距离,用来评价聚类效果)
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心
        # 通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离，距离函数distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist还小,更新minDist和最小质心的index
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index,minDist的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 沿矩阵的列方向计算所有点的均值
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
        print(centroids)
        print(clusterAssment)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment

X_train = np.array(df.loc[:,list(cell_feature)]).T
X_train=np.mat(X_train)

_,cluster=kMeans(X_train,6,distEclud,randCent)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_train)


# In[7]:


# (3) 无监督学习 - 聚类
#  使用sklearn中的GMM
from sklearn.mixture import GaussianMixture
X_train = df.loc[:,list(cell_feature)].to_numpy().T
n_clusters = 6
gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag') # 这里如果用full会爆内存
gmm.fit(X_train)
gmm_labels = gmm.predict(X_train)
print(len(gmm_labels))


# In[8]:


# Calculate NMI and ARI still using sklearn

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
NMI_gmm = normalized_mutual_info_score(ans_labels, gmm_labels)
ARI_gmm = adjusted_rand_score(ans_labels, gmm_labels)
print("NMI for GMM %lf, ARI for GMM %lf" % (NMI_gmm, ARI_gmm))

NMI_kmeans = normalized_mutual_info_score(ans_labels, kmeans_labels)
ARI_kmeans = adjusted_rand_score(ans_labels, kmeans_labels)
print("NMI for kmeans %lf, ARI for kmeans %lf" % (NMI_kmeans, ARI_kmeans))


# In[9]:


#ready for visualize
from sklearn.manifold import TSNE
X = df.loc[:, cell_feature].to_numpy().T
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape


# In[10]:


# plot all the result

# standard answer
color_map = {1:'red', 2:'blue', 3:'green', 4:'yellow', 5:'gray', 6:'pink'}
color_list = [color_map[l] for l in ans_labels]
plt.subplot(141)
plt.scatter(X_embedded[:,0], X_embedded[:,1], color=color_list)
plt.title("Standard answer")

#sklearn_kmeans
kmeans_color_map = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'gray', 5:'pink'}
kmeans_color_list = [kmeans_color_map[i] for i in kmeans.labels_]
plt.subplot(142)
plt.scatter(X_embedded[:,0], X_embedded[:,1], color=kmeans_color_list)
plt.title("kmeans (by library)")

#手写_kMeans
kMeans_color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'gray', 5: 'pink'}
kMeans_color_list = [kMeans_color_map[l[0,0]] for l in cluster[:,0]]
plt.subplot(143)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=kMeans_color_list)
plt.title("kmeans (manual)")

#sklearn_GMM
gmm_color_map = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'gray', 5:'pink'}
gmm_color_list = [gmm_color_map[i] for i in gmm_labels]
plt.subplot(144)
plt.scatter(X_embedded[:,0], X_embedded[:,1], color=gmm_color_list)
plt.title("GMM")


plt.show()


# In[10]:


# (3) 有监督学习
X = df.loc[:,list(cell_feature)].to_numpy().T
y = df_ans.loc[:,'x'].to_numpy()
print(X.shape)
print(y.shape)

# 划分训练集和验证集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=0)

print(X_train.shape)
print(X_test.shape)


# In[104]:


# 手动实现LogisticRegression
print("手动实现的逻辑回归")

theta = np.zeros([X_train.shape[1], 6])
alpha = 0.000001
max_iter = 3001
for this_type in range(6):

    y_train_new = np.where(y_train == (this_type + 1), 1, 0) # 构建该Type下正负样本
    print('训练模型：Type %d' % (1+this_type))
    for step in range(max_iter):
        h = 1 / (1 + np.exp(-np.dot(X_train, theta[:, this_type])))
        theta[:, this_type] = theta[:, this_type] + alpha * np.dot( y_train_new.astype("float64") - h, X_train)
        
        if (step % 500 == 0):
            print("Step = %d, Error=%lf" % ( step , (y_train_new.astype("float64") - h).sum()))
        if (abs( (y_train_new.astype("float64") - h).sum()) < 0.00001):
            break


# In[99]:


# 测试
h = 1 / (1 + np.exp(-np.dot(X_test, theta))) # 计算每种Type的概率
y_predict = h.argmax(axis=1)+1 # 寻找概率的最大值， 需要把+1以把下表转换为type
print("预测", y_predict)
print("答案", y_test)
correct_cnt = np.where(y_test == y_predict, 1, 0).sum()
print("正确率：%lf (%d/%d)" %((correct_cnt/y_test.shape[0]), correct_cnt, y_test.shape[0]))


# In[101]:


# 测试AUC

from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
y_one_hot = label_binarize(y_test,[1,2,3,4,5,6])  #转换成类似二进制的编码
alpha = np.logspace(-2, 2, 20) 
model = LogisticRegressionCV(Cs = alpha, cv = 3, penalty = 'l2')#使用L2正则化 
model.fit(X_train, y_train)
model.C_    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
y_score = model.predict_proba(X_test)
# 1、调用函数计算micro类型的AUC
metrics.roc_auc_score(y_one_hot, y_score, average='micro') 
# 2、手动计算micro类型的AUC 
#首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel())
auc = metrics.auc(fpr, tpr)
print("AUC=%lf"%auc)


# In[ ]:


# 使用python自带的LR

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
score = LR.score(X_test, y_test)
print("使用python自带的逻辑回归")
print("正确率: %lf%%"%(score*100))
print(LR.predict(X_test))
print(y_test)


# In[102]:


# 测试了一下交叉验证比例对于这个正确率的影响

# for percent in range(5, 100, 5):
#     X_train, X_test, y_train, y_test = train_test_split(
#          X, y, test_size=percent/100, random_state=0)
#     LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train, y_train)
#     score = LR.score(X_test, y_test)
#     print("测试集占整个数据集比例: %lf, 正确率 %lf"% (percent/100, score) )

    


# In[ ]:




