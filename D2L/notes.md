# 动手学深度学习 (pytorch)

## 04 数据操作+数据预处理

### pytorch的张量操作



+ 标量

```python
import torch
x = torch.tensor(3.0)
y = torch.tensor(4.0) # x, y都是标量
#对标量进行计算: x + y, x - y, x / y, x**y
```



+ 向量

```python
x = torch.arange(4) 
print(x[2])
print(x.shape)
print(len(x)) # 可以用len访问向量的维度(长度)
```



+ 矩阵，降维

```python
x = torch.arange(20).reshape(5,4)
print(x)
# tensor([[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11],
#       [12, 13, 14, 15],
#       [16, 17, 18, 19]])
print(x.sum(axis=0)) # tensor([40, 45, 50, 55])按哪个轴求和就消去哪个轴
print(x.sum(axis=[0,1])) # 选中所有轴,等价于x.sum()
```



+ 其他张量操作：

```python
x = torch.arange(20.0)
print(x.mean()) # 求平均值, tensor(9.5000)
print(x.sum()/x.numel()) # tensor(9.5000)

x = x.reshape(5,4)
print(x.mean(axis=0)) # tensor([ 8.,  9., 10., 11.])
print(x.sum(axis=0)/x.shape[0]) # tensor([ 8.,  9., 10., 11.]) 
# x.shape[0]: 0维度上有多少个


```



## 自动求导

+ 假如要对$ y = 2x^Tx$ 求导（$x$​是列向量）

```python
import torch
x = torch.arange(4, dtype=float)
x.requires_grad_(True) # 指明x需要存放梯度
y=2*torch.dot(x,x) # y是x的内积,即x^Tx
y.backward() # 求一次导
print(x.grad) # tensor([ 0.,  4.,  8., 12.], dtype=torch.float64)
print(x.grad==4*x) # tensor([True, True, True, True])
```

相当于：

$ x = [x_1,x_2,x_3,x_4]^T$​​

$ y = 2x^Tx = 2(x_1^2+x_2^2+x_3^2+x_4^2) $

$ y'_x = [4x_1, 4x_2, 4x_3, 4x_4] $



假如要对$ z = x_1+x_2+x_3+x_4$求导

```python
x.grad.zero_() # 清除原来的梯度
z = x.sum()
z.backward()
print(x.grad) # tensor([1., 1., 1., 1.], dtype=torch.float64)
```



+ 将某些计算移动到记录的计算图之外



```python
import torch
x = torch.arange(4.0)
x.requires_grad_(True)
y = x*x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad==u)
```

解释：

$ x = [x_1,x_2,x_3,x_4]^T$

$ y = [x_1^2, x_2^2, x_3^2, x_4^2]^T $

$ u = [0^2, 1^2, 2^2, 3^2]^T = [0,1,4,9]^T $

$ z = [0x_1, 1x_2, 4x_3, 9x_4]^T -> z.sum() = 0x_1 + 1x_2 + 4x_3 + 9x_4 $

$ z'_x = [0, 1, 4, 9] $



## 线性回归模拟

训练样本：

$ x=[x_1,x_2,x_3,...,x_n]^T $（其中$x_i$均为向量）

$ y = [y_1, y_2, y_3,...,y_n]^T$

损失函数：$l(X,y,w,b)=\frac {1}{2n}\sum_{i=1}^{n}(y_i-⟨x_i,w〉-b)^2 = \frac {1}{2n}|| Y - Xw -b||^2$

要求对应的$w,b$使得损失最小化

先换元:$X⬅[X,1];w⬅[^w_b]$，则$l(X,y,w)=\frac{1}{2n}||Y-Xw||^2$

损失函数$l$为凸函数，故只需要$l'_w=0$就是要求的$w$



+ 用梯度下降法找$w$

挑选一个初始值$w_0$

重复迭代参数 t = 1, 2, 3

$ w_t = w_{t-1} - η\frac {\partial l}{\partial w_{t-1}} $

$ b_t = b_{t-1} - η\frac {\partial l}{\partial b_{t-1}} $

沿梯度方向将增加损失函数值

学习率η：步长的超参数

步长不能太大，也不能太小



+ 小批量随机梯度下降

在整个训练集上算梯度太贵

随机在n个样本中选取b个样本

SGD: 随机梯度下降，stochastic gradient descent

### 线性回归从零开始实现

+ 自己制造样本集：

```python
def synthetic_data(w, b, numOfExamples):
    X = torch.normal(0,1,size=(numOfExamples,len(w))) # 生成均值为0，方差为1，向量维数为len(w)的numOfExamples个样本（即行为numOfExamples,列为len(w)）
    y = torch.matmul(X,w) + b # torch.matmul()是X和w的向量乘积
    y += torch.normal(0,0.01, y.shape)
    return X, y.reshape(-1,1) # 将y以列向量形式返回

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)
```

+ 用plt画出y与x的第二列的值的关系图：

```python
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
```



+ 将数据打乱并每次随机给batchSize大小的数据：

```python
def data_iter(features,labels,batchSize): # 每次随机抽取大小为batchSize的一批样本
    numOfExamples=len(features)
    indices=list(range(numOfExamples))
    random.shuffle(indices)
    for i in range(0,numOfExamples,batchSize):
        batchIndices=torch.tensor(indices[i:min(i+batchSize,numOfExamples)])
        yield features[batchIndices], labels[batchIndices]
```



+ 定义线性回归方程、均方损失和优化算法（即梯度下降）

```python
def linreg(X, w, b): # 线性回归方程
    return torch.matmul(X,w)+b

def squared_loss(y_hat, y): # 均方损失
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params, lr, batchSize): # 优化算法:每次对params(这里是w和b)中的每一个进行-=lr*param.grad/batchSize,即每次w-=lr*w.grad/batchSize;b-=lr*b.grad/batchSize
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batchSize
            param.grad.zero_()
```



+ 初始化系数w，偏移量b，批大小batchSize，学习率lr，扫描整个数据的次数num_epochs；开始训练

```python
batchSize = 3

w = torch.normal(0,0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)
lr = 0.03
num_epochs = 8 # 把整个数据扫num_epochs遍
net = linreg # 表明神经网络是线性回归模型
loss = squared_loss # 损失函数是平方损失

for epoch in range(num_epochs):
    for X, y in data_iter(features,labels,batchSize):
        l = loss(net(X,w,b), y)
        l.sum().backward() 
        sgd([w, b], lr, batchSize)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')
```

==注意在l.sum().backward()这一步中，梯度w.grad会对批中所有的样本X[i]求导的结果进行累加，故sgd中更新`w=w-w.grad/batchSize`, 因w.grad累积了batchSize次==

## 分类问题与softmax回归

### 分类问题

把n个特征转化为m个输出：独热编码

假设输入为：$ x = [x_1,x_2,x_3,x_4]^T $，输出为$ y = [y_1, y_2, y_3] $

训练数据中y只有一个位置是1，其余位置为0

网络为：

$o_1 = x_1w_{11} + x_2w_{21} + x_3w_{31} + x_4w_{41}$
$o_2 = x_1w_{12} + x_2w_{22} + x_3w_{32} + x_4w_{42}$
$o_3 = x_1w_{13} + x_2w_{23} + x_3w_{33} + x_4w_{43}$

$o = [o_1, o_2, o_3]$

#### softmax函数

$ y = softmax(o) $

$softmax(o) = \frac {e^{o_1}}{\sum_{i=1}^{3}e^{o_i}} \times \frac {e^{o_2}}{\sum_{i=1}^{3}e^{o_i}} \times \frac {e^{o_3}}{\sum_{i=1}^{3}e^{o_i}}$

这样可以确保输出的总和为1，且每个输出都在0到1之间

#### 交叉熵损失函数

交叉熵损失函数：

$l(y,y')=-\sum_{i=1}^{n}y_i\log(y'_i)$

$y$是真实值，$y'$是预测值



### 手写softmax回归

+ 获取数据集

```python
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0)
```



+ 初始化模型参数

```python
num_inputs=28*28
num_outputs=10

W = torch.normal(0,1,(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```



+ 定义softmax函数


```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里用了广播机制
```



+ 定义softmax回归模型

```python
def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs), W) + b)
```




+ 定义交叉熵损失函数

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

```

注意这种写法：（可以通过tensor[[x<sub>1</sub>,x<sub>2</sub>,...,x<sub>n</sub>], [y<sub>1</sub>, y<sub>2</sub>,...,y<sub>n</sub>]]）访问位于[x<sub>i</sub>,y<sub>i</sub>]处的所有元素

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y] # 访问[0,0]和[1,2],即tensor([0.1000, 0.5000])
```

故可以定义交叉熵损失函数



+ 定义模型训练的精度（y_hat中正确预测的用例数量/所有用例数量）

```python
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```



+ 定义累加器类（没什么东西，就是开一个数组然后每次传入一个一样长度的数组，加进来）

```python
class Accumulator:
    # 在n个变量上累加
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```



+ 评估模型的精度函数

```python
def evaluate_accuracy(data_iter, net):
    if isinstance(net, torch.nn.Module): # 如果传入的net是pytorch库里定义好的
        net.eval() # 评估模式, 这会关闭dropout
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad(): # 关闭自动求导
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # y.numel() 返回y的元素个数(numel:num of elements)
    return metric[0] / metric[1]
```



+ 开始训练：

```python
def train_epoch(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()) * len(y), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确度
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(test_iter, net)
        print(f'epoch {epoch + 1}, '
              f'loss {train_loss:.3f}, '
              f'train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')

def updater(batch_size): # 定义自定义的优化器, 这里使用sgd
    return d2l.sgd([W, b], lr, batch_size)
```



## 总结回归类模型

定义损失函数

定义模型，初始化模型

设置好lr, epoch_num, batch_size

进行训练。每个epoch过程：

+ 从data_iter中获取一个批的数据集
+ 计算loss
+ 对loss进行backward()
+ 用求好的grad对参数进行更新
+ 重复epoch



注意在测试模型效果等不需要计算梯度的时候可以用`with torch.no_grad()`来减少开销



## 10. 感知机

### 感知机模型

给出权重w (weight)，偏置b(bias)

$ o = σ(<w,x>+b) $

其中 $ x < 0时 σ(x) = 0, x > 0时σ(x) = 1 $

用于：二分类问题

训练感知机：

```
初始化w = 0, b = 0
若 yi·(<w,x>+b) <= 0, 则更新 w = w + ηxiyi, b = b + ηyi
直到所有训练集正确
```

损失函数：

`loss(y, x, w) = max(0, -y<w,x>)`



### 多层感知机

解决XOR这种无法用单层感知机拟合的

(x > 0 && y > 0) || (x < 0 && y < 0)：输出1

其余情况：输出0

需要用两层感知机，第一层分别判断x和y，第二层根据第一层的两个结果进一步得出最终结果





### 激活函数

不用激活函数会导致整个网络可以转化为一个线性模型

$sigmoid(x) = \frac1{1+e^{-x}} $

$tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}} $

$ ReLU(x) = max(0,x) $ （rectified linear unit）

多类分类：激活函数可用softmax函数

可以多隐藏层，超参数：隐藏层数，每个隐藏层的大小

