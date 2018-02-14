
import DTree_Model as t

filename = input("请输入数据集路径：")   #filename为 DataSet.csv
featureList,labelList = t.loadDataset(filename)   #载入数据集的特征值和类别标签

dummyX,dummyY,vec = t.valueConversion(featureList,labelList)    #数据集转化

clf = t.treesClassification(dummyX,dummyY)  #通过sklearn库创建数据集的决策树

#t.createView(vec,clf)    #创建一个可视化的决策树pdf文件
