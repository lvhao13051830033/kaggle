# coding=utf-8
# 三个库分别是导入分析数据的结果，sklearn库中的逻辑回归，交叉检验
from analyzeData import *
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

'''
    交叉验证，目的应该是为了先判断下分类器的性能怎么样
'''
# 使用分类器类，参数全部默认
classifier = LogisticRegression(C=10000)  # 这个作为5特征分类器，这里的c是正则化系数，防止系统由于过于复杂而产生过拟合
# 调用cross_validation.cross_val_score(分类器，data,target,cv),这个函数返回每次交叉验证的准确率，方法是StratifiedKFold，具体看http://blog.sina.com.cn/s/blog_6a90ae320101a5rc.html
# 函数返回一个数组
score = cross_validation.cross_val_score(classifier, train_five, train_five_label, cv=5)  # 交叉验证
classifier2 = LogisticRegression()  # 这个作为4特征分类器
score2 = cross_validation.cross_val_score(classifier2, train_four, train_four_label, cv=5)  # 交叉验证
print score.mean()

# 下面开始输出结果

#首先处理测试集
testData['Sex']  = testData['Sex'].map({'female': 0, 'male': 1})
testData['Embarked'] = testData['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
testData['family'] = testData['SibSp'] + testData['Parch']
testData['result'] = ''# 这里不能用两个数据集，因为如果将数据分开分别判断，那么无法和乘客号对应
#testData_four = testData[np.isnan(testData['Age'])][['Pclass','Sex','Embarked','family']]
#testData_five = testData.dropna()[['Sex', 'Embarked', 'family', 'Age', 'Pclass']
#读取官方结果]
test_feture = testData[['Sex', 'Embarked', 'family', 'Age', 'Pclass']]
answer = pa.read_csv('gender_submission.csv')

# 训练分类器
classifier.fit(train_five,train_five_label)
classifier2.fit(train_four,train_four_label)

#输出测试集结果
passenger = pa.DataFrame(testData['PassengerId'])
passenger['Survived'] = ''
passengers = passenger.copy()
# isnan()是numpy库里的函数
# # 四个特征的情况
# result_4 = classifier2.predict(testData_four)
# # 五个特征的情况
# result_5 = classifier.predict(testData_five)
# 四特征分类器的输入是只有四个特征的
for i in range(len(testData)):
    if np.isnan(testData['Age'][i]):
        passengers['Survived'][i] = classifier2.predict(testData[['Pclass','Sex','Embarked','family']].ix[i].dropna())
    else:
        passengers['Survived'][i] = classifier.predict(testData[['Sex', 'Embarked', 'family', 'Age', 'Pclass']].ix[i])

# 结算准确率
acc = 1-float(sum(abs(answer['Survived']-passengers['Survived'])))/len(passengers)
print acc