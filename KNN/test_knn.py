import KNN_Model as knn


filename = input("请输入数据集路径:\n")
# filename = irisdata.txt
split = input("请输入数据集分割率:")
columns = input("请输入数据集列数:")
# columns = 5
k = input("请输入最近邻K值:")


filename,split,columns,k = knn.checkParam(filename,split,columns, k)

trainingSet,testSet = knn.loadDataset(filename, split, columns)
print('训练集有: ' + repr(len(trainingSet)) + '个')
print('测试集有: ' + repr(len(testSet)) + '个')


predictions = knn.operatorTestSet(trainingSet,testSet,k)
#predictions1 = knn.operatorTestSet(trainingSet,testSet,20)


accuracy = knn.getAccuracy(testSet, predictions)
#accuracy1 = knn.getAccuracy(testSet, predictions1)
print('预测精度: ' + repr(accuracy) + '%')
#print('预测精度: ' + repr(accuracy1) + '%')

#knn.dataView(trainingSet,testSet)

