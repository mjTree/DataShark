#coding:utf-8
#HierarchicalClustering.py


from numpy import *


"""
function: 定义层次聚类中树节点类
"""
class cluster_node:
    """
    function: 构造函数
    parameter: vec 节点向量
    parameter: left 节点的左儿子
    parameter: right 节点的右儿子
    parameter: distance 与其他节点距离值
    parameter: id 给每个节点分配一个id来判断与其他节点是否计算过距离
    parameter: count 节点个数
    """
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None,count=1):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count
    
    
    """
    function: 计算两个向量之间距离
    parameter: v1 向量1
    parameter: v2 向量2
    """
    def L1dist(v1,v2):
        return sum(abs(v1-v2))
    
    
    """
    function: 计算两个向量之间距离
    parameter: v1 向量1
    parameter: v2 向量2
    """
    def L2dist(v1,v2):
        return sqrt(sum(v1-v2)**2)
    
    
    """
    function: 用于层次聚类
    parameter: features 一个array格式的矩阵
    parameter: distance 距离计算方法
    """
    def hcluster(features,distance=L2dist):
        distances = {}          #存储计算好的distance
        currentclustid = -1     #

        clust = [cluster_node(array(features[i]),id=i) for i in range(len(features))]

        while len(clust)>1:
            lowestpair = (0,1)
            closest = distance(clust[0],vec,clust[1],vec)
            
            for i in range(len(clust)):
                for j in range(i+1,len(clust)):
                    if(clust[i].id,clust[j].id) not in distances:
                        distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)
                    
                    d = distances[(clust[i].id,clust[j].id)]

                    if d<closest:
                        closest = d
                        lowestpair = (i,j)
            
            
            mergeve = [clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i]/2.0 \
                       for i in range(len(clust[0].vec))]

            newcluster = cluster_node(array(mergeve),left=cluster[lowestpair[0]],
                                      right=clust[lowestpair[1]],
                                      distances=closest,id=currentclustid)
            
            currentclustid -= 1
            del clust[lowestpair[1]]
            del clust[lowestpair[0]]
            clust.append(newcluster)
        
        return clust[0]
    
    
    """
    function:
    parameter:
    parameter:
    """
    def extract_clusters(clust,dist):
        clusters = {}
        if clust.distances:
            return [clust]
        else:
            cl = []
            cr = []
            if clust.left != None:
                cl = extract_clusters(clust.left,dist=dist)
            if clust.right != None:
                cl = extract_clusters(clust.right,dist=dist)
            return cl+cr
    
    
    """
    function:
    parameter:
    """
    def get_cluster_elements(clust):
        if clust.id >= 0:
            return [clust.id]
        else:
            cl = []
            cr = []
            if clust.left != None:
                cl = get_cluster_elements(clust.left)
            if clust.right != None:
                cl = get_cluster_elements(clust.right)
            return cl+cr


    """
    function:
    parameter:
    parameter:
    """
    def printclust(clust,labels=None,n=0):
        for i in range(n):
            print ''
        if clust.id<0:
            print '_'
        else:
            if labels == None:
                print clust.id
            else:
                print labels[clust.id]
        
        if clust.left != None:
            printclust(clust.left,labels=labels,n=n+1)
        if clust.right != None:
            printclust(clust.right,labels=labels,n=n+1)
    
    
    """
    function: 获取树的高度
    parameter: clust
    """
    def getheight(clust):
        if clust.left==None and clust.right==None:
            return 1
        return getheight(clust.left)+getheight(clust.right)


    """
    function: 获取树的深度
    parameter: clust
    """
    def getdepth(clust):
        if clust.left==None and clust.right==None:
            return 0
        return max(getdepth(clust.left),getdepth(clust.right))+clust.distance
