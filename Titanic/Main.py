# coding=utf-8
# 三个库分别是导入分析数据的结果，sklearn库中的逻辑回归，交叉检验
from analyzeData import *
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
# 最后一个库是原来cross_validation的替代，原来那个过时了，里边的函数都没变
'''
    交叉验证，目的应该是为了先判断下分类器的性能怎么样
'''
# 使用分类器类
classifier = LogisticRegression(C=10000)  # 这个作为5特征分类器，这里的c是正则化系数，防止系统由于过于复杂而产生过拟合
# 调用cross_validation.cross_val_score(分类器，data,target,cv),这个函数返回每次交叉验证的准确率，方法是StratifiedKFold，具体看http://blog.sina.com.cn/s/blog_6a90ae320101a5rc.html
# 函数返回一个数组
score = model_selection.cross_val_score(classifier, train_five, train_five_label, cv=5)  # 交叉验证
classifier2 = LogisticRegression()  # 这个作为4特征分类器
score2 = model_selection.cross_val_score(classifier2, train_four, train_four_label, cv=5)  # 交叉验证
# 查看平均值
print "五特征分类器交叉验证结果："
print score.mean()
print "四特征分类器交叉验证结果："
print score2.mean()
# 下面开始输出结果

# 首先处理测试集
testData['Sex'] = testData['Sex'].map({'female': 0, 'male': 1})
testData['Embarked'] = testData['Embarked'].map({'C': -1, 'Q': 0, 'S': 1})
testData['family'] = testData['SibSp'] + testData['Parch']
# 这里不能用两个数据集，因为如果将数据分开分别判断，那么无法和乘客号对应
# 读取官方结果
answer = pa.read_csv('gender_submission.csv')

# 训练分类器
classifier.fit(train_five, train_five_label)
classifier2.fit(train_four, train_four_label)

# 输出测试集结果
passengers = pa.DataFrame(testData['PassengerId'])
# 这里必须先创建一个列，直接赋值会出错
passengers['Survived'] = ''
# isnan()是numpy库里的函数
#这里有个问题就是如果直接对passengers['Survived'][i]每行赋值的话会有SettingCopyWarning警告，正确的方法是先生成一个数组temp，然后把数组赋值给列，直接添加会有警告
#这里创建数组不能用np.zeros(),因为用zeros生成的是float数组，最后结果要求int类型
temp = [0] * passengers.shape[0]
for i in range(len(testData)):
    if np.isnan(testData['Age'][i]):
        # 这里也有一个问题：由于需要一个一个做预测，但是predict函数要求输入二维数组，单行数据是一维的
        #所以要把数据转为numpy中的array，然后转为二维数组
        #reshape()函数可以利用原数组数据改变数组的尺寸，但是主要新旧数组数据量相同，而reshape(1,-1)可以将[xxx]转为[[xxx]]这种，就可以用了
        temp[i] = int(classifier2.predict(np.array( testData[['Pclass', 'Sex', 'Embarked', 'family']].ix[i].dropna()).reshape(1,-1)))
    else:
        temp[i] = int(classifier.predict(np.array(testData[['Sex', 'Embarked', 'family', 'Age', 'Pclass']].ix[i]).reshape(1,-1)))
print temp
passengers['Survived'] = temp
# 结算准确率
acc = 1 - float(sum(abs(answer['Survived'] - passengers['Survived']))) / len(passengers)
print "准确率是："
print acc
# 输出文件
passengers.to_csv('result.csv',encoding='utf-8',index=False)
