#coding:utf-8
#SVM_Model.py

import numpy as np
from sklearn import svm
from sklearn import preprocessing
import csv, sys
import random


"""
function: 加载数据集,将数据集分成4个部分
parameter: filename 数据集路径
parameter: split 数据集分割率
return: trainingSet 训练集特征值    testSet 测试集特征值
        trainingLabel 训练集类标签  testLabel 训练集类标签
"""
def loadDataset(filename, split):
    trainingSet = []    #训练集特征值
    trainingLabel = []  #训练集标签
    testSet = []        #测试集特征值
    testLabel = []      #测试集标签
    
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        header = next(lines)    #得一行数据来确定数据集列数
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(len(header)-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingLabel.append(dataset[x][-1])
                trainingSet.append(dataset[x][:-1])
            else:
                testLabel.append(dataset[x][-1])
                testSet.append(dataset[x][:-1])
    return trainingSet,testSet,trainingLabel,testLabel



"""
function: 用于将字符类型的类标签转化为SKL库中SVM所需的格式
parameter: labelList 存放类标签的字符元素列表
return: xL 用数字表示的类标签列表
"""
def valueConversion(labelList):
    lb = preprocessing.LabelBinarizer()
    xlabel = lb.fit_transform(labelList)
    xL = []    #存储用数字表示类标签的列表
    for i in range(len(xlabel)):
        total = 0
        for y in range(len(xlabel[0])):
            total += xlabel[i][y]*(y+1)
        xL.append(total)
    return xL



"""
function: 调用SKL中的svm得到超平面来分类数据集
parameter: dataSet 数据集特征值列表
parameter:labelSet 类标签列表
return: clf 通过训练集得到的分类器
"""
def svmClassification(dataSet,labelSet):
    clf = svm.SVC(kernel = 'linear')
    clf.fit(dataSet,labelSet)
    return clf



"""
function: 通过训练集得到的分类器在测试集上做测试
parameter: testSet 测试集数据
parameter: testLabel 测试集类标签
parameter: clf SVM分类器
"""
def getAccuracy(testSet,testLabel,clf):
    right = 0    #存放测试集正确个数
    for i in range(len(testSet)):
        if testLabel[i] == clf.predict([testSet[i]]):
            right += 1
    return(right/float(len(testLabel)))*100.0
