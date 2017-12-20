# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 首先处理数据

# 读取文件
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
answerData = pd.read_csv('sampleSubmission.csv')

# 处理训练数据
# 提取出时间作为一个特征
temp = [0]*trainData.shape[0]
for i in range(trainData.shape[0]):
    temp[i] = int(trainData['datetime'][i].split()[1].split(':')[0])
trainData['hour'] = temp

temp = [0]*testData.shape[0]
for i in range(testData.shape[0]):
    temp[i] = int(testData['datetime'][i].split()[1].split(':')[0])
testData['hour'] = temp

# 分开数据和标记
trainY = trainData['count']
#这里注意drop是返回一个新数组，原数组操作后没有改变
trainX = trainData.drop(['casual','registered','datetime','count','atemp'],axis=1)
# # 交叉验证
# # 回归器，参数：具体的看文档，一般就是设置最大迭代次数n_estimators，还要设置随机数种子random_state
learner = RandomForestRegressor(n_estimators=130,random_state=0)
# score = cross_val_score(learner,trainX,trainY,cv = 5)
# print score.mean()
learner.fit(trainX,trainY)
result = learner.predict(testData.drop(['datetime','atemp'],axis=1))
answerData['count'] = result.astype(int)
answerData.to_csv('result.csv',index=False)
