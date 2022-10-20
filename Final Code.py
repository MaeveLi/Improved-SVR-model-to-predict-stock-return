# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:56:11 2020

@author: Mae
"""

#######基于支持向量回归的股票收益率预测模型 python代码############



######################################第二章代码############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
pd.set_option('display.max_columns', None)

#读入数据
JPM0 = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\JPM.csv')
JPM0.head()
JPM = JPM0.set_index('date').copy() #把date作为了index
JPM = JPM.iloc[0:396]

#处理数据
logreturn = JPM.nextlogreturn
logreturn = np.array(logreturn)
logreturn.shape 
logreturndf = pd.DataFrame(logreturn)
logreturndf.describe() #转换成数据框才能用describe
JPM1 = JPM.drop('nextlogreturn', axis = 1)    
JPM1.shape  
JPM1.head()

####变量相关性
sns.pairplot(JPM1) #加载太久，不直观
JPM1.corr()
sns.heatmap(JPM1.corr())

###############以第一个训练集为例子展示PCA的提取效果
firsttrain = JPM1.iloc[0:378]
firsttest = JPM1.iloc[378:396]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]

ss1=StandardScaler()
firsttrainst = ss1.fit_transform(firsttrain)
np.mean(firsttrainst),np.std(firsttrainst)
dftrain = pd.DataFrame(firsttrainst)

firsttestst = ss1.transform(firsttest)
np.mean(firsttestst),np.std(firsttestst)
dftest = pd.DataFrame(firsttestst)

ss2 = StandardScaler()
y_train = ss2.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 

y_test = ss2.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 
st_ytest = pd.DataFrame(y_test)

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, firsttrainst.shape[0]+1)).T  
nmtest = np.mat(range(1, firsttestst.shape[0]+1)).T 

##########训练集提取
pca_x = PCA(n_components=12)
pcaxtrain= pca_x.fit_transform(dftrain)
pcaxtrain_df = pd.DataFrame(data = pcaxtrain, 
                            columns = ['principal component 1', 'principal component 2'
                                       , 'principal component 3', 'principal component 4'
                                       , 'principal component 5', 'principal component 6'
                                       , 'principal component 7', 'principal component 8'
                                       , 'principal component 9', 'principal component 10'
                                       , 'principal component 11', 'principal component 12'])
print(pcaxtrain_df)
print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))

#scree plot 碎石图
plt.figure()
plt.plot(range(1,13), pca_x.explained_variance_ratio_ , c='royalblue', linewidth=2, 
         marker='o')
plt.xlabel('principal components', fontsize=16)
plt.ylabel('explained variance', fontsize=16)
for i in range(1,13):
    plt.text(i, pca_x.explained_variance_ratio_[i-1], 
             '{:.4f}'.format(pca_x.explained_variance_ratio_[i-1])) #加文字
plt.title('Scree Plot')
plt.grid(linestyle='-.')
plt.show()

#累计图
accu_var=np.cumsum(pca_x.explained_variance_ratio_)
accu_var=np.insert(accu_var,0,0)
plt.figure()
plt.plot(range(0,13), accu_var, c='royalblue', 
         linewidth=2, marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.ylim(-0.01,1)
for i in range(0,13):
    plt.text(i, accu_var[i]-0.02, '{:.4f}'.format(accu_var[i]))
plt.title('accumulated explained variance')
plt.grid(linestyle='-.')
plt.axhline(y=0.95,ls="--",c="burlywood")   #加一条直线
plt.show()

#确定主成分个数为8后重新提取主成分到数据框中
pca_x = PCA(n_components = 8)
pcaxtrain= pca_x.fit_transform(dftrain)
pcaxtrain_df = pd.DataFrame(data = pcaxtrain, 
                            columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
#各成分载荷:Multiply each component by the square root of its corresponding eigenvalue:
loadings=pd.DataFrame(data=pca_x.components_.T * np.sqrt(pca_x.explained_variance_),
                      columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])#得到载荷矩阵
loadings #查看载荷
loadings.to_csv(r'E:\【Mae】\毕设\实验结果\载荷矩阵.csv', index = False) #保存到文件
#再次查看热力图
sns.heatmap(pcaxtrain_df.corr())

#################以第一个训练集为例子展示KPCA的提取效果
def stepwise_kpca(X, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: 数组形式的M*N的矩阵，M为样本数量，N为特征个数
        gamma: RBF核的调优参数
        n_components: 返回的主成分的数量

    """
    # 计算数据集中成对的欧几里得距离
    sq_dists = pdist(X, 'sqeuclidean')

    #将成对距离转换为方阵
    mat_sq_dists = squareform(sq_dists)

    #计算核矩阵
    K = exp(-gamma * mat_sq_dists)

    #中心化核矩阵
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) #K是一个N*N的矩阵，进一步说明了在核主成分分析中，主成分的个数是由样本数量决定的

    #从核矩阵中得到特征值与特征向量（降序）
    eigvals, eigvecs = eigh(K)

    #得到最高的n个特征值对应的特征向量。与标准主成分不同，这里的特征向量不是主成分轴，而是将样本映射到这些轴上的主成分。
    #即跳过了pca所有的需要特征矩阵转换数据的步骤，因此也不能表现出载荷与主成分的意义
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    
    # 修改处，收集相应的特征值
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]
    
    #查看有多少个大于1的特征值，计算特征值之和。这一步必须是提取全部的378个主成分才能运行的,要改下面的代码，把#号删掉，并将之前的return前加上#号
    #eig1=0
    #k=0
    #for i in range(0,378):
    #    if lambdas[i]>1:
    #        eig1+=lambdas[i]
    #       k=k+1
    #return X_pc,lambdas[0],np.sum(lambdas),eig1/np.sum(lambdas),k 
    #(这一步仅用于kpca效果验证，需要验证查看效果时去掉#、删掉下一行即可)
    return X_pc,lambdas

stepwise_kpca(dftrain,0.009,18)   #可以通过改变gamma值与将18改成378得到很多结果 结合上面改了的函数
kpcatrain, lambdas = stepwise_kpca(dftrain,0.009,18) #将返回的主成分与特征值分别赋值
kpcatraindf = pd.DataFrame(data = kpcatrain,
                           columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
                                      'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16',
                                      'PC17', 'PC18'])
#查看热力图
sns.heatmap(kpcatraindf.corr())

#画特征值递减图    
plt.figure()
plt.plot(range(1,19), lambdas , c='royalblue', linewidth=2, 
         marker='o')
plt.xlabel('kernel principal components', fontsize=12)
plt.ylabel('eigenvalues', fontsize=12)
for i in range(1,19):
    plt.text(i, lambdas[i-1], 
             '{:.2f}'.format(lambdas[i-1]))
plt.title('KPCA eigenvalues',fontsize=16)
plt.grid(axis='y',linestyle='-.')
plt.show()

#累计图
kpcatrain, lambdas = stepwise_kpca(dftrain,0.009,36)
kpca_explained_variance_ratio=[]
for i in range(36):
    kpca_explained_variance_ratio.append(lambdas[i]/141.84) 
accu_var=np.cumsum(kpca_explained_variance_ratio)
accu_var=np.insert(accu_var,0,0)

plt.figure()
plt.plot(range(0,37), accu_var, c='royalblue', 
         linewidth=2, marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.ylim(-0.01,1)
for i in range(0,31,5):
    plt.text(i, accu_var[i]-0.02, '{:.4f}'.format(accu_var[i]))
plt.title('kpca accumulated explained variance')
plt.grid(linestyle='-.')
plt.axhline(y=0.95,ls="--",c="burlywood")
plt.show()




#############################################第三章代码##################################################
#本章额外需要的一些函数定义
import mplfinance as mpf
from matplotlib.pyplot import MultipleLocator
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
def DS(pred,n):
    upup=0
    updown=0
    downup=0
    downdown=0
    for i in range(n):  #根据不同的问题 有不同的range(n) n为预测集天数 需要临时调整并重新定义函数
        if logreturntest[i]>0 and pred[i]>0:
            upup+=1
        elif logreturntest[i]>0 and pred[i]<=0:
            updown+=1
        elif logreturntest[i]<=0 and pred[i]>0:
            downup+=1
        else:
                    downdown+=1
    print(upup,updown,downup,downdown)



##################中国市场：招商银行########################
CMB0 = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\招商银行.csv')
CMB0.head()
CMB0 = CMB0.iloc[0:396]
date=CMB0.date
CMB = CMB0.set_index('date').copy() 

#描述统计
CMB['nextlogreturn'].describe() 

#candlestick graph/k线图 这一部分可能是因为包的原因 每次只能画一个图就会卡 要关掉整个console重新画第二个
CMBmini = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\CMBmini.csv')
CMBmini.dtypes
CMBmini.Date=pd.to_datetime(CMBmini.Date)
CMBmini = CMBmini.set_index('Date').copy()
CMBmini.index.name = 'Date'
mc = mpf.make_marketcolors(up='white',down='g',volume={'up':'r','down':'g'},
                           edge={'up':'r','down':'g'},wick={'up':'r','down':'g'}) #in,i,inh都能代表inherit价格的配色
selfmade = mpf.make_mpf_style(base_mpf_style='yahoo',                             
                              marketcolors=mc,
                              y_on_right=False,
                              rc = {'axes.spines.bottom':True, 
                                          'axes.spines.left':True, 
                                          'axes.spines.right':True, 
                                          'axes.spines.top':True, 
                                          'xtick.color':'black', 
                                          'ytick.color':'black', 
                                          'axes.labelcolor':'black' }) #自定义了一个主题
mpf.plot(CMBmini.iloc[0:396],type='candle',style=selfmade,title='CMB Candlestick Chart',mav=(3,8,12),volume=True) 
mpf.plot(CMBmini.iloc[378:396],type='candle',style=selfmade,title='CMB Test Set Candlestick Chart',mav=(3,8,12),volume=True) 
#最平缓的是mav9，反之为mav3

#对数据框进行一些编辑调整
logreturn = CMB.nextlogreturn
logreturn = np.array(logreturn)
logreturn.shape 
logreturndf = pd.DataFrame(logreturn)
logreturndf.describe() #转换成数据框才能用describe
CMB1 = CMB.drop('nextlogreturn', axis = 1)    
CMB1.shape  
CMB1.head()

#对收益率进行画图
x_major_locator=MultipleLocator(30)  #设置坐标轴主要刻度的间隔
_, ax = plt.subplots(figsize=[14, 7])
ax.plot(date, logreturn, c='royalblue', label='logreturn')
ticks = ax.get_xticks()
plt.xticks(rotation=-15)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel('date')
plt.ylabel('value')
plt.grid(linestyle='-.',)
plt.axhline(y=0,ls="-",c="brown")
plt.title('CMB Log Return')
plt.legend()
plt.show()

##########简单svr##########
#Split
CMB1train = CMB1.iloc[0:378, ]
CMB1test = CMB1.iloc[378:396, ]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]

#Scale
ss_x = StandardScaler()
x_train = ss_x.fit_transform(CMB1train)
np.mean(x_train), np.std(x_train) 
x_test = ss_x.transform(CMB1test)
np.mean(x_test), np.std(x_test) 

ss_y = StandardScaler()
y_train = ss_y.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 
y_test = ss_y.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, x_train.shape[0]+1)).T  
nmtest = np.mat(range(1, x_test.shape[0]+1)).T 

#Regression
svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)
rbf = svr_rbf.fit(x_train, y_train) 
trainpred = svr_rbf.predict(x_train)  
testpred = svr_rbf.predict(x_test)
testpred0=ss_y.inverse_transform(testpred)
print("简单rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,
                                                ss_y.inverse_transform(testpred)))
print("简单rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,
                                                ss_y.inverse_transform(testpred)))
DS(testpred0,18)

#画训练集的图
_, ax = plt.subplots(figsize=[14, 7])
ax.plot(np.array(nmtrain), np.array(ss_y.inverse_transform(y_train)), c='red', label='data')
ax.plot(nmtrain,ss_y.inverse_transform(trainpred), c='green', label='SVR Model')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.title('CMB simple SVR training set')
plt.grid(linestyle='-.',)
plt.legend()
plt.show()
#画测试集的图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(ss_y.inverse_transform(y_test)), c='#F61909', label='data',s=20)
ax.plot(nmtest, ss_y.inverse_transform(testpred), c='#486D0B', label='SVR Model')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('CMB simple SVR test set')
plt.legend()
plt.show()

#######滑窗svr########
values=0
for i in range(0,16,3):
    ###Split data
    CMB1train = CMB1.iloc[i:i+378, ]
    CMB1test = CMB1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, CMB1train.shape[0]+1)).T  
    nmtest = np.mat(range(1, CMB1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(CMB1train)#fit_transform先拟合数据 再将其转换为标准模式 一般用于训练集 因为训练集我们找得到均值和方差
    np.mean(x_train), np.std(x_train) #查看标准化后的数据的均值和方差
    x_test = ss_x.transform(CMB1test)
    np.mean(x_test), np.std(x_test) 
    
    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    np.mean(y_train), np.std(y_train)     
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    np.mean(y_test), np.std(y_test) 
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    ###simple svr
    svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)  
    simplerbf = svr_rbf.fit(x_train, y_train) 
    strainpred = svr_rbf.predict(x_train)  
    stestpred = svr_rbf.predict(x_test)
    if i==0:
        values=ss_y.inverse_transform(stestpred)
    else:
            values=np.hstack((values,ss_y.inverse_transform(stestpred))) 
###模型评价
#MSE,MAE
logreturntest = logreturn[378:396]
print("滑窗rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,values))
print("滑窗rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,values))
DS(values,18)
#对比有无滑窗的合并图
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest, testpred0, c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest, values, c='royalblue', label='moving forward SVR Model',marker='o')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('CMB SVR test set comparison')
plt.legend()
plt.show()

#############PCA############
firsttrain = CMB1.iloc[0:378]
firsttest = CMB1.iloc[378:396]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]

ss1=StandardScaler()
firsttrainst = ss1.fit_transform(firsttrain)
np.mean(firsttrainst),np.std(firsttrainst)
dftrain = pd.DataFrame(firsttrainst)

firsttestst = ss1.transform(firsttest)
np.mean(firsttestst),np.std(firsttestst)
dftest = pd.DataFrame(firsttestst)

ss2 = StandardScaler()
y_train = ss2.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 

y_test = ss2.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 
st_ytest = pd.DataFrame(y_test)

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, firsttrainst.shape[0]+1)).T  
nmtest = np.mat(range(1, firsttestst.shape[0]+1)).T 

##########训练集提取
pca_x = PCA(n_components=12)
pcaxtrain= pca_x.fit_transform(dftrain)
pcaxtrain_df = pd.DataFrame(data = pcaxtrain, 
                            columns = ['principal component 1', 'principal component 2'
                                       , 'principal component 3', 'principal component 4'
                                       , 'principal component 5', 'principal component 6'
                                       , 'principal component 7', 'principal component 8'
                                       , 'principal component 9', 'principal component 10'
                                       , 'principal component 11', 'principal component 12'])
print(pcaxtrain_df)
print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))

#scree plot 碎石图
plt.figure()
plt.plot(range(1,13), pca_x.explained_variance_ratio_ , c='royalblue', linewidth=2, 
         marker='o')
plt.xlabel('principal components', fontsize=16)
plt.ylabel('explained variance', fontsize=16)
for i in range(1,13):
    plt.text(i, pca_x.explained_variance_ratio_[i-1], 
             '{:.4f}'.format(pca_x.explained_variance_ratio_[i-1]))
plt.title('Scree Plot')
plt.grid(linestyle='-.')
plt.show()

#累计图
accu_var=np.cumsum(pca_x.explained_variance_ratio_)
accu_var=np.insert(accu_var,0,0)
plt.figure()
plt.plot(range(0,13), accu_var, c='royalblue', 
         linewidth=2, marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.ylim(-0.01,1)
for i in range(0,13):
    plt.text(i, accu_var[i]-0.02, '{:.4f}'.format(accu_var[i]))
plt.title('accumulated explained variance')
plt.grid(linestyle='-.')
plt.axhline(y=0.95,ls="--",c="burlywood")
plt.show() 

#确定主成分个数为7
pca_x = PCA(n_components=7)
pcaxtrain= pca_x.fit_transform(dftrain)
pcaxtrain_df = pd.DataFrame(data = pcaxtrain, 
                            columns = ['principal component 1', 'principal component 2'
                                       , 'principal component 3', 'principal component 4'
                                       , 'principal component 5', 'principal component 6'
                                       , 'principal component 7'])
pcaxtest= np.dot(dftest,pca_x.components_.T)
pcaxtest_df = pd.DataFrame(data = pcaxtest, 
                            columns = ['principal component 1', 'principal component 2'
                                       , 'principal component 3', 'principal component 4'
                                       , 'principal component 5', 'principal component 6'
                                       , 'principal component 7'])
############PCA+SVR模型
svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.4)
rbf = svr_rbf.fit(pcaxtrain, y_train) 
spcatrainpred = svr_rbf.predict(pcaxtrain)  
spcatestpred = svr_rbf.predict(pcaxtest)
print("简单pca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,
                                                ss2.inverse_transform(spcatestpred)))
print("简单pca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,
                                                ss2.inverse_transform(spcatestpred)))
logreturntest = logreturn[378:396]
DS(spcatestpred,18)
#对比图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest, testpred0, c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest, ss2.inverse_transform(spcatestpred), c='royalblue', label='PCA-SVR Model',marker='x')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('CMB PCA-SVR test set comparison')
plt.legend()
plt.show()

###########KPCA############
firsttrain = CMB1.iloc[0:378]
firsttest = CMB1.iloc[378:396]

ss1=StandardScaler()
firsttrainst = ss1.fit_transform(firsttrain)
np.mean(firsttrainst),np.std(firsttrainst)
dftrain = pd.DataFrame(firsttrainst)

firsttestst = ss1.transform(firsttest)
np.mean(firsttestst),np.std(firsttestst)
dftest = pd.DataFrame(firsttestst)

stepwise_kpca(dftrain,0.01,18)
kpcatrain, lambdas = stepwise_kpca(dftrain,0.01,18)
kpcatraindf = pd.DataFrame(data = kpcatrain,
                           columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
                                      'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16',
                                      'PC17', 'PC18'])
#画特征值递减图    
plt.figure()
plt.plot(range(1,19), lambdas , c='royalblue', linewidth=2, 
         marker='o')
plt.xlabel('kernel principal components', fontsize=12)
plt.ylabel('eigenvalues', fontsize=12)
for i in range(1,19):
    plt.text(i, lambdas[i-1], 
             '{:.2f}'.format(lambdas[i-1]))
plt.title('KPCA eigenvalues',fontsize=16)
plt.grid(axis='y',linestyle='-.')
plt.show()
#累计图
kpcatrain, lambdas = stepwise_kpca(dftrain,0.01,36)
kpca_explained_variance_ratio=[]
for i in range(36):
    kpca_explained_variance_ratio.append(lambdas[i]/153.36) 
accu_var=np.cumsum(kpca_explained_variance_ratio)
accu_var=np.insert(accu_var,0,0)

plt.figure()
plt.plot(range(0,37), accu_var, c='royalblue', 
         linewidth=2, marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.ylim(-0.01,1)
for i in range(0,31,3):
    plt.text(i, accu_var[i]-0.02, '{:.4f}'.format(accu_var[i]))
plt.title('kpca accumulated explained variance')
plt.grid(linestyle='-.')
plt.axhline(y=0.95,ls="--",c="burlywood")
plt.show()

###############kpca+svr
kpcatrain, lambdastrain = stepwise_kpca(dftrain,0.01,18)
kpcatest, lambdastest = stepwise_kpca(dftest,0.01,18)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0002)
rbf = svr_rbf.fit(kpcatrain, y_train) 
kpcatrainpred = svr_rbf.predict(kpcatrain)  
kpcatestpred = svr_rbf.predict(kpcatest)
print("简单kpca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396],
                                                ss2.inverse_transform(kpcatestpred)))
print("简单kpca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                ss2.inverse_transform(kpcatestpred)))
DS(kpcatestpred,18)

#对比图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest, testpred0, c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest, ss2.inverse_transform(kpcatestpred), c='royalblue', label='KPCA-SVR Model',marker='x')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('CMB KPCA-SVR test set comparison')
plt.legend()
plt.show()

##############滑窗svr+PCA####################
pcavalues=0
for i in range(0,16,3):
    ###Split data
    CMB1train = CMB1.iloc[i:i+378, ]
    CMB1test = CMB1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, CMB1train.shape[0]+1)).T 
    nmtest = np.mat(range(1, CMB1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(CMB1train)
    st_xtrain = pd.DataFrame(x_train, columns=CMB1.columns) 
    x_test = ss_x.transform(CMB1test)
    st_xtest = pd.DataFrame(x_test, columns=CMB1.columns) 
    
    ss_y = StandardScaler()
    y_train0 = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    ytrain=y_train0[0:375]
    yvali=y_train0[375:378]
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    
    y_train0 = y_train0.flatten()
    ytrain = ytrain.flatten()
    yvali = yvali.flatten()
    y_test = y_test.flatten()
    nmtrain = np.mat(range(1, x_train.shape[0]+1)).T 
    nmtest = np.mat(range(1, x_test.shape[0]+1)).T 
    
    ##pca
    pca_x = PCA(n_components = 3)
    pcaxtrain0= pca_x.fit_transform(x_train)
    pcaxtrain = pcaxtrain0[0:375]
    pcaxvali = pcaxtrain0[375:378]
    
    pcaxtest = np.dot(x_test,pca_x.components_.T)
    
    #grid search
    print(i+1)
    bestmae = 1
    for gamma in  [0.35,0.37,0.39,0.4,0.43,0.45]:
        for C in [800,900,1000,1100,1200]: #如果要把参数调优去掉，可以把上面的数组删到只剩0.4，下面的只剩1000，即可
            gridsvr = SVR(gamma=gamma,C=C)
            gridsvr.fit(pcaxtrain,ytrain)
            valipred=gridsvr.predict(pcaxvali)
            gridmae=mean_absolute_error(ss_y.inverse_transform(yvali),ss_y.inverse_transform(valipred))
            if gridmae < bestmae:
                bestmae = gridmae
                best_parameters = {'gamma':gamma,'C':C}
    print("Best parameters:{}".format(best_parameters))
    print("Best mae:{}".format(bestmae))
    
    ###pca+svr
    svr_rbf = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])  
    simplerbf = svr_rbf.fit(pcaxtrain0, y_train0) 
    pcatrainpred = svr_rbf.predict(pcaxtrain0)  
    pcatestpred = svr_rbf.predict(pcaxtest)
    if i==0:
        pcavalues=ss_y.inverse_transform(pcatestpred)
    else:
            pcavalues=np.hstack((pcavalues,ss_y.inverse_transform(pcatestpred))) #每一次的标准不一样 不能最后一起inverse transform
print("滑窗pca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396], 
                                                ss_y.inverse_transform(pcavalues)))
print("滑窗pca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                ss_y.inverse_transform(pcavalues)))
logreturntest = logreturn[378:396]
DS(pcavalues,18)
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 

#KPCA滑窗
kpcavalues=0
for i in range(0,16,3):
    ###Split data
    CMB1train = CMB1.iloc[i:i+378, ]
    CMB1test = CMB1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, CMB1train.shape[0]+1)).T 
    nmtest = np.mat(range(1, CMB1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(CMB1train)
    st_xtrain = pd.DataFrame(x_train, columns=CMB1.columns) 
    x_test = ss_x.transform(CMB1test)
    st_xtest = pd.DataFrame(x_test, columns=CMB1.columns) 
    
    ss_y = StandardScaler()
    y_train0 = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    ytrain=y_train0[0:375]
    yvali=y_train0[375:378]
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    
    y_train0 = y_train0.flatten()
    ytrain = ytrain.flatten()
    yvali = yvali.flatten()
    y_test = y_test.flatten()
    nmtrain = np.mat(range(1, x_train.shape[0]+1)).T 
    nmtest = np.mat(range(1, x_test.shape[0]+1)).T 
    
    ##kpca
    kpcaxtrain0=stepwise_kpca(x_train,0.01,3)[0] 
    kpcaxtrain=kpcaxtrain0[0:375]
    kpcavali=kpcaxtrain0[375:378]
    kpcaxtest= stepwise_kpca(x_test,0.01,3)[0]
    
    #grid search
    print(i+1)
    bestmae = 1
    for gamma in [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007]:
        for C in [500,800,1000,1100,1200,1500]: #如果要把参数调优去掉，可以把上面的数组删到只剩0.0002，下面的只剩1000，即可
            gridsvr = SVR(gamma=gamma,C=C)
            gridsvr.fit(kpcaxtrain,ytrain)
            valipred=gridsvr.predict(kpcavali)
            gridmae=mean_absolute_error(ss_y.inverse_transform(yvali),ss_y.inverse_transform(valipred))
            if gridmae < bestmae:
                bestmae = gridmae
                best_parameters = {'gamma':gamma,'C':C}
    print("Best parameters:{}".format(best_parameters))
    print("Best mae:{}".format(bestmae))
    
    ###pca+svr
    svr_rbf = SVR(kernel='rbf',  C=best_parameters['C'], gamma=best_parameters['gamma'])  #不考虑epsilon 只对C和gamma进行参数寻优？
    kpca_rbf = svr_rbf.fit(kpcaxtrain, ytrain) 
    kpcatrainpred = svr_rbf.predict(kpcaxtrain)  
    kpcatestpred = svr_rbf.predict(kpcaxtest)
    if i==0:
        kpcavalues=ss_y.inverse_transform(kpcatestpred)
    else:
            kpcavalues=np.hstack((kpcavalues,ss_y.inverse_transform(kpcatestpred))) #每一次的标准不一样 不能最后一起inverse transform
print("滑窗kpca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396],
                                                kpcavalues))
print("滑窗kpca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                kpcavalues))
logreturntest = logreturn[378:396]
DS(kpcavalues,18)
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 



###########################美国###############################
#读入数据
JPM0 = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\JPM.csv')
JPM = JPM0.iloc[0:396]
date = JPM.date
JPM = JPM.set_index('date').copy() 

#描述统计   
JPM['nextlogreturn'].describe()      

#candlestick graph
JPMmini = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\JPMmini.csv')
JPMmini.dtypes
JPMmini.Date=pd.to_datetime(JPMmini.Date)
JPMmini = JPMmini.set_index('Date').copy()
JPMmini.index.name = 'Date'
mc = mpf.make_marketcolors(up='white',down='g',volume={'up':'r','down':'g'},
                           edge={'up':'r','down':'g'},wick={'up':'r','down':'g'})
selfmade = mpf.make_mpf_style(base_mpf_style='yahoo',                             
                              marketcolors=mc,
                              y_on_right=False,
                              rc = {'axes.spines.bottom':True, 
                                          'axes.spines.left':True, 
                                          'axes.spines.right':True, 
                                          'axes.spines.top':True, 
                                          'xtick.color':'black', 
                                          'ytick.color':'black', 
                                          'axes.labelcolor':'black' }) 
mpf.plot(JPMmini.iloc[0:396],type='candle',style=selfmade,title='JPM Candlestick Chart',mav=(3,8,12),volume=True) 
mpf.plot(JPMmini.iloc[378:396],type='candle',style=selfmade,title='JPM Test Set Candlestick Chart',mav=(3,8,12),volume=True) 
#最平缓的是mav12，反之为mav3

#对数据框进行一些编辑调整
logreturn = JPM.nextlogreturn
logreturn = np.array(logreturn)
logreturndf = pd.DataFrame(logreturn)
logreturndf.describe() #转换成数据框才能用describe
JPM1 = JPM.drop('nextlogreturn', axis = 1)    

#对收益率进行画图
x_major_locator=MultipleLocator(30)  #设置坐标轴主要刻度的间隔
_, ax = plt.subplots(figsize=[14, 7])
ax.plot(date, logreturn, c='royalblue', label='logreturn')
ticks = ax.get_xticks()
plt.xticks(rotation=-15)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlabel('date')
plt.ylabel('value')
plt.grid(linestyle='-.',)
plt.axhline(y=0,ls="-",c="brown")
plt.title('JPM Log Return')
plt.legend()
plt.show()

####滑窗简单svr
for i in range(0,16,3):
    ###Split data
    JPM1train = JPM1.iloc[i:i+378, ]
    JPM1test = JPM1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, JPM1train.shape[0]+1)).T  
    nmtest = np.mat(range(1, JPM1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(JPM1train)#fit_transform先拟合数据 再将其转换为标准模式 一般用于训练集 因为训练集我们找得到均值和方差
    np.mean(x_train), np.std(x_train) #查看标准化后的数据的均值和方差
    x_test = ss_x.transform(JPM1test)
    np.mean(x_test), np.std(x_test) 
    
    ss_y = StandardScaler()
    y_train = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    np.mean(y_train), np.std(y_train)     
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    np.mean(y_test), np.std(y_test) 
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    ###simple svr
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  
    simplerbf = svr_rbf.fit(x_train, y_train) 
    strainpred = svr_rbf.predict(x_train)  
    stestpred = svr_rbf.predict(x_test)
    if i==0:
        values=ss_y.inverse_transform(stestpred)
    else:
            values=np.hstack((values,ss_y.inverse_transform(stestpred))) #每一次的标准不一样 分别inverse transform

####不划窗svr
#Split
JPM1train = JPM1.iloc[0:378, ]
JPM1test = JPM1.iloc[378:396, ]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]
#Scale
ss_x = StandardScaler()
x_train = ss_x.fit_transform(JPM1train)#fit_transform先拟合数据 再将其转换为标准模式 一般用于训练集 因为训练集我们找得到均值和方差
np.mean(x_train), np.std(x_train) #查看标准化后的数据的均值和方差
x_test = ss_x.transform(JPM1test)
np.mean(x_test), np.std(x_test) 

ss_y = StandardScaler()
y_train = ss_y.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 
y_test = ss_y.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, x_train.shape[0]+1)).T  
nmtest = np.mat(range(1, x_test.shape[0]+1)).T 

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
rbf = svr_rbf.fit(x_train, y_train) 
trainpred = svr_rbf.predict(x_train)  
testpred = svr_rbf.predict(x_test)
###模型评价
#MSE,MAE
print("滑窗rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,values))
print("简单rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,
                                                ss_y.inverse_transform(testpred)))
print("滑窗rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,values))
print("简单rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,
                                                ss_y.inverse_transform(testpred)))
DS(testpred,18)
DS(values,18)
####Plot
#画训练集的图
_, ax = plt.subplots(figsize=[14, 7])
ax.plot(np.array(nmtrain), np.array(ss_y.inverse_transform(y_train)), c='red', label='data')
ax.plot(nmtrain,ss_y.inverse_transform(trainpred), c='green', label='SVR Model')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM simple SVR training set')
plt.legend()
plt.show()
#画测试集的图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(ss_y.inverse_transform(y_test)), c='#F61909', label='data',s=20)
ax.plot(nmtest, ss_y.inverse_transform(testpred), c='#486D0B', label='SVR Model')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM simple SVR test set')
plt.legend()
plt.show()
#合并图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest, ss_y.inverse_transform(testpred), c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest, values, c='royalblue', label='moving forward SVR Model',marker='o')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM SVR test set comparison')
plt.legend()
plt.show()

################普通svr+pca###################
firsttrain = JPM1.iloc[0:378]
firsttest = JPM1.iloc[378:396]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]

ss1=StandardScaler()
firsttrainst = ss1.fit_transform(firsttrain)
np.mean(firsttrainst),np.std(firsttrainst)
dftrain = pd.DataFrame(firsttrainst)

firsttestst = ss1.transform(firsttest)
np.mean(firsttestst),np.std(firsttestst)
dftest = pd.DataFrame(firsttestst)

ss2 = StandardScaler()
y_train = ss2.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 

y_test = ss2.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 
st_ytest = pd.DataFrame(y_test)

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, firsttrainst.shape[0]+1)).T  
nmtest = np.mat(range(1, firsttestst.shape[0]+1)).T 

pca_x = PCA(n_components = 8)
pcaxtrain= pca_x.fit_transform(dftrain)
pcaxtest= np.dot(dftest,pca_x.components_.T)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.3)
rbf = svr_rbf.fit(pcaxtrain, y_train) 
spcatrainpred = svr_rbf.predict(pcaxtrain)  
spcatestpred = svr_rbf.predict(pcaxtest)
print("简单pca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturntest,
                                                ss2.inverse_transform(spcatestpred)))
print("简单pca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturntest,
                                                ss2.inverse_transform(spcatestpred)))
logreturntest = logreturn[378:396]
DS(spcatestpred,18)
spcavalues=ss2.inverse_transform(spcatestpred)

#对比图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest,ss_y.inverse_transform(testpred), c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest,spcavalues, c='royalblue', label='PCA-SVR Model',marker='x')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM PCA-SVR test set comparison')
plt.legend()
plt.show()

#################普通svr+kpca#####################
#kpca+svr
kpcatrain, lambdastrain = stepwise_kpca(dftrain,0.009,18)
kpcatest, lambdastest = stepwise_kpca(dftest,0.009,18)
svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.001)
rbf = svr_rbf.fit(kpcatrain, y_train) 
kpcatrainpred = svr_rbf.predict(kpcatrain)  
kpcatestpred = svr_rbf.predict(kpcatest)
print("简单kpca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396],
                                                ss2.inverse_transform(kpcatestpred)))
print("简单kpca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                ss2.inverse_transform(kpcatestpred)))
DS(kpcatestpred,18)
skpcavalues=ss2.inverse_transform(kpcatestpred)

#对比图
_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest,ss_y.inverse_transform(testpred), c='k', label='simple SVR Model',marker='o')
ax.plot(nmtest,skpcavalues, c='royalblue', label='KPCA-SVR Model',marker='x')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM KPCA-SVR test set comparison')
plt.legend()
plt.show()

###############滑窗SVR+kpca###################
kpcavalues=0
for i in range(0,16,3):
    ###Split data
    JPM1train = JPM1.iloc[i:i+378, ]
    JPM1test = JPM1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, JPM1train.shape[0]+1)).T 
    nmtest = np.mat(range(1, JPM1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(JPM1train)
    st_xtrain = pd.DataFrame(x_train, columns=JPM1.columns) 
    x_test = ss_x.transform(JPM1test)
    st_xtest = pd.DataFrame(x_test, columns=JPM1.columns) 
    
    ss_y = StandardScaler()
    y_train0 = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    ytrain=y_train0[0:375]
    yvali=y_train0[375:378]
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    
    y_train0 = y_train0.flatten()
    ytrain = ytrain.flatten()
    yvali = yvali.flatten()
    y_test = y_test.flatten()
    nmtrain = np.mat(range(1, x_train.shape[0]+1)).T 
    nmtest = np.mat(range(1, x_test.shape[0]+1)).T 
    
    ##kpca
    kpcaxtrain0=stepwise_kpca(x_train,0.009,3)[0] 
    kpcaxtrain=kpcaxtrain0[0:375]
    kpcavali=kpcaxtrain0[375:378]
    kpcaxtest= stepwise_kpca(x_test,0.009,3)[0]
    
    #grid search
    print(i+1)
    bestmae = 1
    for gamma in [0.0005,0.001,0.002,0.003,0.004,0.005,0.007]:
        for C in [300,500,800,1000,1100,1200,1500]:  #如果只需要查看滑窗效果 删到gamma只有0.001，C只有1000即可
            gridsvr = SVR(gamma=gamma,C=C)
            gridsvr.fit(kpcaxtrain,ytrain)
            valipred=gridsvr.predict(kpcavali)
            gridmae=mean_absolute_error(ss_y.inverse_transform(yvali),ss_y.inverse_transform(valipred))
            if gridmae < bestmae:
                bestmae = gridmae
                best_parameters = {'gamma':gamma,'C':C}
    print("Best parameters:{}".format(best_parameters))
    print("Best mae:{}".format(bestmae))
    
    ###pca+svr
    svr_rbf = SVR(kernel='rbf',  C=best_parameters['C'], gamma=best_parameters['gamma'])  #不考虑epsilon 只对C和gamma进行参数寻优？
    kpca_rbf = svr_rbf.fit(kpcaxtrain, ytrain) 
    kpcatrainpred = svr_rbf.predict(kpcaxtrain)  
    kpcatestpred = svr_rbf.predict(kpcaxtest)
    if i==0:
        kpcavalues=ss_y.inverse_transform(kpcatestpred)
    else:
            kpcavalues=np.hstack((kpcavalues,ss_y.inverse_transform(kpcatestpred))) #每一次的标准不一样 不能最后一起inverse transform
print("滑窗kpca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396],
                                                kpcavalues))
print("滑窗kpca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                kpcavalues))
logreturntest = logreturn[378:396]
DS(kpcavalues,18)
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 

##############滑窗svr+PCA####################
pcavalues=0
for i in range(0,16,3):
    ###Split data
    JPM1train = JPM1.iloc[i:i+378, ]
    JPM1test = JPM1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, JPM1train.shape[0]+1)).T 
    nmtest = np.mat(range(1, JPM1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(JPM1train)
    st_xtrain = pd.DataFrame(x_train, columns=JPM1.columns) 
    x_test = ss_x.transform(JPM1test)
    st_xtest = pd.DataFrame(x_test, columns=JPM1.columns) 
    
    ss_y = StandardScaler()
    y_train0 = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    ytrain=y_train0[0:375]
    yvali=y_train0[375:378]
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    
    y_train0 = y_train0.flatten()
    ytrain = ytrain.flatten()
    yvali = yvali.flatten()
    y_test = y_test.flatten()
    nmtrain = np.mat(range(1, x_train.shape[0]+1)).T 
    nmtest = np.mat(range(1, x_test.shape[0]+1)).T 
    
    ##pca
    pca_x = PCA(n_components = 18)
    pcaxtrain0= pca_x.fit_transform(x_train)
    pcaxtrain = pcaxtrain0[0:375]
    pcaxvali = pcaxtrain0[375:378]
    
    pcaxtest = np.dot(x_test,pca_x.components_.T)
    
    #grid search
    print(i+1)
    bestmae = 1
    for gamma in [0.15,0.18,0.2,0.23,0.25,0.28,0.3]:
        for C in [1000,1200,1500]:
            gridsvr = SVR(gamma=gamma,C=C)
            gridsvr.fit(pcaxtrain,ytrain)
            valipred=gridsvr.predict(pcaxvali)
            gridmae=mean_absolute_error(ss_y.inverse_transform(yvali),ss_y.inverse_transform(valipred))
            if gridmae < bestmae:
                bestmae = gridmae
                best_parameters = {'gamma':gamma,'C':C}
    print("Best parameters:{}".format(best_parameters))
    print("Best mae:{}".format(bestmae))
    
    ###pca+svr
    svr_rbf = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])  
    simplerbf = svr_rbf.fit(pcaxtrain0, y_train0) 
    pcatrainpred = svr_rbf.predict(pcaxtrain0)  
    pcatestpred = svr_rbf.predict(pcaxtest)
    if i==0:
        pcavalues=ss_y.inverse_transform(pcatestpred)
    else:
            pcavalues=np.hstack((pcavalues,ss_y.inverse_transform(pcatestpred))) #每一次的标准不一样 不能最后一起inverse transform
print("滑窗pca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396], 
                                                pcavalues))
print("滑窗pca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                pcavalues))
logreturntest = logreturn[378:396]
DS(pcavalues,18)
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 

npcavalues=0
for i in range(0,16,3):
    ###Split data
    JPM1train = JPM1.iloc[i:i+378, ]
    JPM1test = JPM1.iloc[i+378:i+381, ]
    logreturntrain = logreturn[i:i+378]
    logreturntest = logreturn[i+378:i+381]
    nmtrain = np.mat(range(1, JPM1train.shape[0]+1)).T 
    nmtest = np.mat(range(1, JPM1test.shape[0]+1)).T 
    
    ###Scale data
    ss_x = StandardScaler()
    x_train = ss_x.fit_transform(JPM1train)
    st_xtrain = pd.DataFrame(x_train, columns=JPM1.columns) 
    x_test = ss_x.transform(JPM1test)
    st_xtest = pd.DataFrame(x_test, columns=JPM1.columns) 
    
    ss_y = StandardScaler()
    y_train0 = ss_y.fit_transform(logreturntrain.reshape(-1,1))
    ytrain=y_train0[0:375]
    yvali=y_train0[375:378]
    y_test = ss_y.transform(logreturntest.reshape(-1,1))
    
    y_train0 = y_train0.flatten()
    ytrain = ytrain.flatten()
    yvali = yvali.flatten()
    y_test = y_test.flatten()
    nmtrain = np.mat(range(1, x_train.shape[0]+1)).T 
    nmtest = np.mat(range(1, x_test.shape[0]+1)).T 
    
    ##pca
    pca_x = PCA(n_components = 18)
    pcaxtrain0= pca_x.fit_transform(x_train)
    pcaxtrain = pcaxtrain0[0:375]
    pcaxvali = pcaxtrain0[375:378]
    
    pcaxtest = np.dot(x_test,pca_x.components_.T)
    
    #grid search
    print(i+1)
    bestmae = 1
    for gamma in [0.3]:
        for C in [1000]:
            gridsvr = SVR(gamma=gamma,C=C)
            gridsvr.fit(pcaxtrain,ytrain)
            valipred=gridsvr.predict(pcaxvali)
            gridmae=mean_absolute_error(ss_y.inverse_transform(yvali),ss_y.inverse_transform(valipred))
            if gridmae < bestmae:
                bestmae = gridmae
                best_parameters = {'gamma':gamma,'C':C}
    print("Best parameters:{}".format(best_parameters))
    print("Best mae:{}".format(bestmae))
    
    ###pca+svr
    svr_rbf = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])  
    simplerbf = svr_rbf.fit(pcaxtrain0, y_train0) 
    pcatrainpred = svr_rbf.predict(pcaxtrain0)  
    pcatestpred = svr_rbf.predict(pcaxtest)
    if i==0:
        npcavalues=ss_y.inverse_transform(pcatestpred)
    else:
            npcavalues=np.hstack((npcavalues,ss_y.inverse_transform(pcatestpred))) #每一次的标准不一样 不能最后一起inverse transform
print("滑窗pca+rbf核函数支持向量机的MSE为:", mean_squared_error(logreturn[378:396], 
                                                npcavalues))
print("滑窗pca+rbf核函数支持向量机的MAE为:", mean_absolute_error(logreturn[378:396],
                                                npcavalues))
logreturntest = logreturn[378:396]
DS(npcavalues,18)
nmtest = np.mat(range(1, logreturntest.shape[0]+1)).T 

####不划窗svr
#Split
JPM1train = JPM1.iloc[0:378, ]
JPM1test = JPM1.iloc[378:396, ]
logreturntrain = logreturn[0:378]
logreturntest = logreturn[378:396]
#Scale
ss_x = StandardScaler()
x_train = ss_x.fit_transform(JPM1train)#fit_transform先拟合数据 再将其转换为标准模式 一般用于训练集 因为训练集我们找得到均值和方差
np.mean(x_train), np.std(x_train) #查看标准化后的数据的均值和方差
x_test = ss_x.transform(JPM1test)
np.mean(x_test), np.std(x_test) 

ss_y = StandardScaler()
y_train = ss_y.fit_transform(logreturntrain.reshape(-1,1))
np.mean(y_train), np.std(y_train) 
y_test = ss_y.transform(logreturntest.reshape(-1,1))
np.mean(y_test), np.std(y_test) 

y_train = y_train.flatten()
y_test = y_test.flatten()
nmtrain = np.mat(range(1, x_train.shape[0]+1)).T  
nmtest = np.mat(range(1, x_test.shape[0]+1)).T 

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
rbf = svr_rbf.fit(x_train, y_train) 
trainpred = svr_rbf.predict(x_train)  
testpred = svr_rbf.predict(x_test)

_, ax = plt.subplots(figsize=[14, 7])
ax.scatter(np.array(nmtest), np.array(logreturntest), c='red', label='original data',s=15)
ax.plot(nmtest, pcavalues, c='royalblue', label='moving forward grid search PCA+SVR Model',marker='x')
ax.plot(nmtest, npcavalues, c='g', label='moving forward PCA+SVR Model',marker='x')
ax.plot(nmtest, ss_y.inverse_transform(testpred), c='k', label='simple SVR Model',marker='o')
ticks = ax.get_xticks()
plt.xlabel('data')
plt.ylabel('target')
plt.grid(linestyle='-.',)
plt.title('JPM PCA+SVR test set comparison')
plt.legend()
plt.show()



##############################################第四章###################################################
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

###########JPM银行###########
JPM0 = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\JPMmini2.csv')
JPM0.head()
JPM = JPM0.iloc[0:378]
returns = pd.Series(JPM['logreturn'].values, index=JPM['Date'])
prices = pd.Series(JPM['Close'].values, index=JPM['Date'])
X=returns.values
P=prices.values

###收益率
#单位根检验
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))  #p值为0 认为该收益率序列平稳
#ACF检验
plot_acf(X)
plt.show()
#PACF检验
plot_pacf(X, lags=50)
plt.show()
#检验白噪声序列
acorr_ljungbox(X,lags=40)#是白噪声序列，停止预测

###价格
#单位根检验
result = adfuller(P)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) #略有一些不平稳，进行一阶差分
#一阶差分+单位根检验
prices1=prices.diff()
prices1=prices1.drop(['2018/2/1'],axis=0)
P1=prices1.values
result1 = adfuller(P1)
print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])
print('Critical Values:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value)) #p值为0，序列平稳
#检验白噪声序列
acorr_ljungbox(P1,lags=20)#是白噪声序列，停止预测

###对数价格
#取对数
P.shape #(378,)
for i in range(378):
    P[i]=math.log(P[i])
P
#ADF检验
result = adfuller(P)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) #略有一些不平稳，进行一阶差分
#一阶差分后ADF检验
prices1=prices.diff()
prices1=prices1.drop(['2018/2/1'],axis=0)
P1=prices1.values
result1 = adfuller(P1)
print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])
print('Critical Values:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value)) #p值为0，序列平稳
#检验白噪声序列
acorr_ljungbox(P1,lags=40)#是白噪声序列，停止预测


#########招商银行#########
CMB0 = pd.read_csv('E:\\【Mae】\\毕设\\数据\\数据\\CMBmini2.csv')
CMB0.head()
CMB = CMB0.iloc[0:378]
returns = pd.Series(CMB['logreturn'].values, index=CMB['Date'])
prices = pd.Series(CMB['close'].values, index=CMB['Date'])
X=returns.values
P=prices.values

###收益率
#单位根检验
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))  #p值为0 认为该收益率序列平稳
#ACF检验
plot_acf(X)
plt.show()
#PACF检验
plot_pacf(X, lags=50)
plt.show()
#检验白噪声序列
acorr_ljungbox(X,lags=40) #是白噪声

###价格
#单位根检验/adf
result = adfuller(P)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) 
#一阶差分后adf
prices1=prices.diff()
prices1=prices1.drop(['2018/2/1'],axis=0)
P1=prices1.values
result1 = adfuller(P1)
print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])
print('Critical Values:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value)) 
#检验白噪声序列
acorr_ljungbox(P1,lags=20) #是白噪声
LB=acorr_ljungbox(P1,lags=20)
LB_df = pd.DataFrame(data = LB, 
                            columns = ['Q-statistic', 'p-value'])
#or
r, q, p = sm.tsa.acf(P1, qstat=True)
data = np.c_[range(1, 21), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
print(table.set_index('lag'))
table.to_csv(r'E:\【Mae】\毕设\实验结果\白噪声检验.csv', index = True)
#ACF检验
plot_acf(X)
plt.show()
#PACF检验
plot_pacf(X, lags=50)
plt.show()

###对数价格
P.shape #(378,)
for i in range(378):
    P[i]=math.log(P[i])
P
#ADF检验
result = adfuller(P)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) #略有一些不平稳，进行一阶差分
#一阶差分后ADF检验
prices1=prices.diff()
prices1=prices1.drop(['2018/2/1'],axis=0)
P1=prices1.values
result1 = adfuller(P1)
print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])
print('Critical Values:')
for key, value in result1[4].items():
	print('\t%s: %.3f' % (key, value)) #p值为0，序列平稳
#检验白噪声序列
acorr_ljungbox(P1,lags=40)
