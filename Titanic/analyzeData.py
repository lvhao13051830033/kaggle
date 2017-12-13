# coding=utf-8

import numpy as np
import pandas as pa
import matplotlib.pyplot as pl

'''
    处理数据部分主要分为：
    1.画图判断哪些特征是对结果没有影响的，不用这些特征值训练
    2.数据预处理，把可以合并的特征合并，不是数值类型的转化为数值类型，处理缺失数据
    3.根据挑选出的特征建立数据集和标记数组
'''
# pandas库里的read_csv可以把csv文件读取为dataFrame对象，这个对象就是很多列（数组）共享一个index，这个index 是真的存在的且可以指定，每一列都有name
# 其形状大小是真实数据的形状，不包括index和name
# 下边两句分别适用的绝对路径和相对路径，注意路径中的反斜线是两根，或者一根/
train_obj = pa.read_csv('train.csv')
test_obj = pa.read_csv('test.csv')

# 把觉得对结果没有影响的数据去掉
# DataFrame对象的drop方法：删除行或者列，可以多行，drop（'行名或者列名'，axis=0/1(0是行，1是列)），多个行名或者列名：['','']
trainData = train_obj.drop(['Name', 'Ticket', 'Cabin'], axis=1)
testData = test_obj.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 下面用画图的形式对其他变量对最后结果的影响作出判断：
# 首先考察船舱和性别，用groupby函数对的数据进行分类，groupby（'列名'），根据给出的列，把具有相同数值的合为一组，组数为列值的种数
# 如果groupby（['列名1'，'列名2']），则列1为分大类，大类中再根据列2进行分类
train_group1 = trainData.groupby(['Sex', 'Pclass'])  # 有6组
rate = (train_group1.sum() / train_group1.count())['Survived']  # 有6组，每组都只有一个值，生存率

# 下面开始画图，柱状图
x = np.array([1, 2, 3])
wi = 0.3
# bar(left，height,width,color)用来化柱状图,参数分别是每个柱子的左侧位置坐标，高度（一般就是数据），宽度，颜色,前三个一般都是集合
pl.bar(x, rate.female, wi, color='r')
pl.bar(x, rate.male, wi, color='b')
pl.title("survived rate")
pl.xlabel("Pclass")
pl.ylabel("rate")
# 指定网格线格式
pl.grid(True, linestyle='-', color='0.7')
# 轴的刻度
pl.xticks([1, 2, 3])
# arange(起点，终点，间隔)，创建一个等差序列
pl.yticks(np.arange(0.0, 1.1, 0.1))
# 对不同的柱做标记
pl.legend(['female', 'male'])
#pl.show()

# 接下来的过程和上边差不多，分别是判断了年龄、上船地点、船上亲属人数

# 船上的亲属人数就是把两个代表家属的变量相加
trainData['family'] = trainData['SibSp'] + trainData['Parch']
# 复制一个数据集，因为一会要两个分类器
trainData2 = trainData.copy()
# pandas中的map()函数一般是用来进行数据转换的，参数是函数或者字典，一个series调用map(),会对照字典用的key，将series中的key全部改成相应的value,map()函数返回一个新的，对源对象不操作
trainData2['Sex'] = trainData2['Sex'].map({'female': 0, 'male': 1})
trainData2['Embarked'] = trainData2['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# numpy.isnan()函数的作用是返回一个Boolean数组，对应input的下标，若当前位置是nan那么就是true，否则false
# 挑选出age为nan的，这个集合不用age这个特征
train_age_nan = trainData2[np.isnan(trainData2['Age'])]
# 挑选出age不为空的数据，直接调用dropna(),这个方法会去掉输入数组中有nan特征的数据，挑选后的具备所有特征，所以可以用所有特征
train_all = trainData2.dropna()

# 对于以上两个数据集分别挑选出需要的特征组成训练集，因为之前已经确认了有的是对结果没有影响的，所以可以只挑选有用的
#dataframe选取列的格式是两个方括号，[['列名','列名']]
train_five = train_all[['Sex', 'Embarked', 'family', 'Age', 'Pclass']]
train_four = train_age_nan[['Sex', 'Embarked', 'family', 'Pclass']]

# 两个数据集的标记数组
train_five_label = train_all['Survived']
train_four_label = train_age_nan['Survived']

# 至此处理数据工作全部完成，接下来就是训练模型
