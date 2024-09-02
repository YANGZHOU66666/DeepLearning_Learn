import matplotlib.pyplot as plt
import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, numOfExamples):
    X = torch.normal(0,1,size=(numOfExamples,len(w))) # 生成均值为0，方差为1，向量维数为len(w)的numOfExamples个样本（即行为numOfExamples,列为len(w)）
    y = torch.matmul(X,w) + b 
    y += torch.normal(0,0.01, y.shape)
    return X, y.reshape(-1,1) # 将y以列向量形式返回

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)


"""
x2 = features[:, 1].detach().numpy()  # 第二个特征维度
y = labels[:, 0].detach().numpy()     # 标签

# 创建散点图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.scatter(x2, y, cmap='viridis')  # 绘制散点图，c=y用于根据标签值着色

# 设置图表标题和坐标轴标签
plt.xlabel('Feature 2')
plt.ylabel('Label')
plt.title('Synthetic Data') 

# 显示图形
plt.show() 
"""

def data_iter(features,labels,batchSize): # 每次随机抽取大小为batchSize的一批样本
    numOfExamples=len(features)
    indices=list(range(numOfExamples))
    random.shuffle(indices)
    for i in range(0,numOfExamples,batchSize):
        batchIndices=torch.tensor(indices[i:min(i+batchSize,numOfExamples)])
        yield features[batchIndices], labels[batchIndices]

batchSize = 10

w = torch.normal(0,0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b): # 线性回归方程
    return torch.matmul(X,w)+b


def squared_loss(y_hat, y): # 均方损失,注意这里y_hat和y都是向量,维数取决于batch_size
    return (y_hat-y.reshape(y_hat.shape))**2 / 2


# sgd: stochastic gradient descent 随机梯度下降
def sgd(params, lr, batchSize): # 优化算法:每次对params(这里是w和b)中的每一个进行-=lr*param.grad/batchSize,即每次w-=lr*w.grad/batchSize;b-=lr*b.grad/batchSize
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batchSize  # 注意这里param.grad是累加的,所以要除以batchSize才是平均值
            param.grad.zero_()

lr = 0.03
num_epochs = 8 # 把整个数据扫num_epochs遍
net = linreg # 表明神经网络是线性回归模型
loss = squared_loss # 损失函数是平方损失

for epoch in range(num_epochs):
    for X, y in data_iter(features,labels,batchSize):
        l = loss(net(X,w,b), y) 
        l.sum().backward()  # 计算梯度,注意这里l.sum()是对batchSize个样本求和,所以要除以batchSize才是平均值
        sgd([w, b], lr, batchSize)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')