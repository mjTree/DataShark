#coding:utf-8
#NeuralNetwork.py


import numpy as np


"""
function: 定义双曲线型tanh的函数
parameter: x 一个值
return: 对应的tanh双曲线函数值
"""
def tanh(x):
    return np.tanh(x)


"""
function: 定义双曲线型tanh的导函数
parameter: x 一个值
return: 对应的tanh双曲线函数导数值
"""
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


"""
function: 定义S型曲线logistic函数
parameter: x 一个值
return: 对应的S型曲线逻辑函数值
"""
def logistic(x):
    return 1/(1 + np.exp(-x))


"""
function: 定义S型曲线logistic的导函数
parameter: x 一个值
return: 对应的S型曲线逻辑函数导数值
"""
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))


"""
describe: 设计一个神经网络的类
"""
class NeuralNetwork:
    """
    function: 构造函数,初始化类对象
    parameter: layers 一个list,长度为神经网络的层数,内容为每层的神经元个数
    parameter: activation 选取logistic函数/tanh函数
    """
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # 随机生成权重值
        self.weights = []
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)  #与前一层连线权重分配
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)    #与后一层连线权重分配
            #print(str(self.weights))
    
    
    """
    function: 训练神经网络
    parameter: X 训练集
    parameter: y 类标签集
    parameter: learning_rate 学习率
    parameter: epochs 循环次数
    """
    def fit(self, X, y, learning_rate, epochs):
        X = np.atleast_2d(X)                        #确认X至少二维矩阵
        temp = np.ones([X.shape[0], X.shape[1]+1])  #初始化矩阵全是1(行列数+1是为了有B这个偏向)
        temp[:, 0:-1] = X                           #行全选，第一列到倒数第二列
        X = temp
        y = np.array(y)  #数据结构转换
        
        for k in range(epochs):                     #抽样梯度下降epochs抽样
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            for l in range(len(self.weights)):
                #向前传播,得到每个节点的输出结果
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            error = y[i] - a[-1]    #最后一层错误率
            deltas = [error * self.activation_deriv(a[-1])]
            
            for l in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
    
    
    """
    function: 用于预测数据类别
    parameter: x 测试的数据值
    return: a 预测值
    """
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
