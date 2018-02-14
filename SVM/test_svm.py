import SVM_Model as svm
import pylab as pl
import numpy as np

filename = input("请输入数据集路径：")    # irisdata.txt
split = input("请输入数据集分割率：")
trainingSet,testSet,trainingLabel,testLabel = svm.loadDataset(filename,float(split))

print('训练集有: ' + repr(len(trainingSet)) + '个')
print('测试集有: ' + repr(len(testSet)) + '个')

# 转化类标签表示方式
trainingLabel = svm.valueConversion(trainingLabel)
testLabel = svm.valueConversion(testLabel)

clf = svm.svmClassification(trainingSet, trainingLabel)    #通过训练集得到svm分类器

accuracy = svm.getAccuracy(testSet,testLabel,clf)   #用测试集对分类器做测试
print('预测精度: ' + repr(accuracy) + '%')
