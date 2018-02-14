#coding:utf-8
#KMeans.py

import numpy as np


"""
function: 
parameter: X 数据集
parameter: k k个类别类数目
parameter: maxIt 最大迭代次数
return: dataSet 数据集
"""
def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape    #行数、列数
    
    dataSet = np.zeros((numPoints, numDim+1))
    dataSet[:,:-1] = X
    
    centroids = dataSet[np.random.randint(numPoints, size=k), :]   #随机选取K个中心点
    #centroids = dataSet[0:2, :]    #强制选取前两个点做初始点
    
    centroids[:, -1] = range(1, k+1)
        
    iterations = 0    #迭代次数
    oldCentroids = None
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        #print "iterations:\n",iterations
        #print "dataSet:\n",dataSet
        #print "centroids:\n",centroids
        
        oldCentroids = np.copy(centroids)
        iterations += 1
        
        updateLabels(dataSet, centroids)
        
        centroids = getCentroids(dataSet, k)
    return dataSet


"""
function: 检测聚类是否该停止
parameter: oldCentroids 上一轮中心点
parameter: centroids 本轮中心点
parameter: iterations 当前迭代次数
parameter: maxIt 最大迭代次数
return: true/false 是否停止
"""
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)


"""
function: 将数据集归类到中心点
parameter: dataSet 数据集
parameter: centroids 中心点
"""
def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape
    for i in range(0,numPoints):
        dataSet[i, -1] = getLabelFromCloseCentroid(dataSet[i, :-1], centroids)


"""
function: 将数据点归类到一个中心点
parameter: dataSetRow 待归类的数据点
parameter: centroids 中心点
return: label 该数据点所属的类别
"""
def getLabelFromCloseCentroid(dataSetRow, centroids):
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    #print "minDist:",minDist
    return label


"""
function: 重新计算更新后的数据集中心点
parameter: dataSet 划分好的数据集
parameter: k k个类别数目
return: result 更新后的中心点
"""
def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k+1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i
    return result


"""
function: 做测试
"""
def main():
    x1 = np.array([1,1])
    x2 = np.array([2,1])
    x3 = np.array([4,3])
    x4 = np.array([5,4])
    testX = np.vstack((x1, x2, x3, x4))

    result = kmeans(testX, 2, 10)
    print "final result:\n",result
    #print "final result:\n",result[:, -1]

main()
