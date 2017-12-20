#coding=utf-8

import numpy as np
import pandas as pd
# 构造一个DataFrame：pd.DataFrame({'key1':数组1，'key2':数组2})
a = pd.DataFrame({'col1': [1, 2,3], 'col2': [3, 4,5]})
print a
# DataFrame.values会按行返回数据，每行是一个数组，最后结果是一个二维数组
print a.values[:,0]
# 生成数组可以用下边的方式：循环体放在前边，后边是for语句
print [i for i in range(10)]
print a['col1'][1]
# shape()返回一个元祖，第一个元素是行数，第二个元素是列数
print a.shape

b = '2011/1/1  1:00:00'
print b.split()[1].split(':')[0]