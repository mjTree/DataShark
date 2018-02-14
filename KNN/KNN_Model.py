#coding:utf-8
#KNN_Model.py

import sys
import csv
import random
import math
import operator


"""
function: 加载数据集文件
parameter: filename 数据集路径文件名
parameter: split 分割数据集来创建训练集和测试集
parameter: columns 数据集有列数
return: trainingSet 训练集
return: testSet 测试集
"""
def loadDataset(filename, split,columns):
    trainingSet = []    #训练集
    testSet = []        #测试集
    
    #dataset = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(int(columns)-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    return trainingSet,testSet


"""
function: 计算距离
parameter: instance1 事例1
parameter: instance2 事例2
parameter: length 事例的维度
return: euclideanDistance 返回euclidean的距离
"""
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)


"""
function: 获取最近的K个相邻数
parameter: trainingSet 训练集
parameter: testInstance 测试集中一个事例
parameter: k K值
return: neighbors K个最近的邻居
"""
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        #distances.append(dist)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors


"""
function: 统计最近的K个邻居属于那些类别
parameter: neighbors K个最近的邻居
return: sortedVotes[0][0] 最相似的类别
"""
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


"""
function: 对分类结果进行检测
parameter: testSet 测试集
parameter: predictions 测试集真实类别
return: 精度值
"""
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


"""
function: 对输入参数进行判断
parameter: fiename 文件路径
parameter: split 分割数据集比例
parameter: columns 数据集的列数
parameter: k 最近邻的K值
return: 
"""
def checkParam(filename, split, columns, k):
    if(filename == ''):
        print('数据集不存在,请重新读取！！')
        sys.exit()
    if(columns == ''):
        print('抱歉,你没有输入数据集列数！！')
        sys.exit()
    else:
        columns = int(columns)
    if(split == ''):
        float(split = 0.7)  #默认分割0.7
    else:
        split = float(split)
    if(k == ''):
        int(k = 4)      #默认K值为4
    else:
        k = int(k)
    return filename,split,columns,k

