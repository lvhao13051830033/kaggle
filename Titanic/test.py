#coding=utf-8

import numpy as np
import pandas as pd

a = np.array([1,2,3, np.NaN])
#numpy数组可以用一个boolean数组来选择元素，对应位置是True就是选出来
b = a[[True,False,True,True]]
# 创建DataFrame：pd.DataFrame(直接给出一个字典，字典的key是每列的名字，value是一个数组)
# 第二种方式是：参数直接给出一个二维数组，然后给出index=[],columns=[]，分别是行号和列名
dataFrame = pd.DataFrame({'id':[1,2,3,4,5],'age':[12,5,69,45,33]})
# dropna()是删除有nan数据的行
# ix[i]是第i行的数据，格式是：
'''
    列名 数据
    列名 数据
    列名 数据
 '''
# 所以对ix[]使用dropna()的话，相当于删除了这行中有nan数据的那列数据
dataFrame_fature = dataFrame
dataFrame_fature.ix[0] = dataFrame[['id']].ix[0]
print dataFrame_fature
print dataFrame_fature.dropna()
print dataFrame_fature.ix[0]
print dataFrame_fature.ix[0].dropna()

dataFrame2 = pd.DataFrame({'id':[1,2,3,4,5],'age':[12,5,69,np.nan,33]})
print dataFrame2.dropna()
print dataFrame2['id']
print dataFrame2.ix[1]