# Deep-ML刷题记录

#### 2025.7.27

- 22. Sigmoid

注意python的exp为：

```python
import math
math.exp(x)
```

保留k位小数：

```
round(x,k)
```

- 23. Softmax

- 24. Single Neuron + MSE

注意MSE的计算方法：

对n个预测标签$\{\hat{y_i}, i∈[1,n]\}$及其标签$\{y_i,i∈[1,n]\}$，$MSE = \frac{1}{n}\sum(y_i-\hat{y_i})^2$



#### 2025.7.28

- 25. Single Neuron + MSE + Back Propagation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(features, weights, bias):
    return np.dot(features, weights) + bias

def MSE(predictions, labels):
    return np.mean((predictions - labels) ** 2)

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    features = np.array(features, dtype=np.float64)         # shape: (n_samples, n_features)
    labels = np.array(labels, dtype=np.float64)             # shape: (n_samples,)
    weights = np.array(initial_weights, dtype=np.float64)   # shape: (n_features,)
    bias = float(initial_bias)
    mse_values = []

    for epoch in range(epochs):
        # Forward pass
        logits = linear(features, weights, bias)             # shape: (n_samples,)
        predictions = sigmoid(logits)                        # shape: (n_samples,)
        loss = MSE(predictions, labels)
        mse_values.append(round(loss, 4))

        # Backward pass
        # Gradie
        dL_dp = 2 * (predictions - labels)                   # shape: (n_samples,)
        dp_dz = predictions * (1 - predictions)              # shape: (n_samples,)
        dL_dz = dL_dp * dp_dz                                # shape: (n_samples,)

        # Gradients
        grad_w = np.dot(dL_dz, features) / len(features)     # shape: (n_features,)
        grad_b = np.mean(dL_dz)

        # Parameter update
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias, mse_values
```

- key points:

numpy.array的一些用法：

```python
import numpy as np
arr = [1.0,2.0,3.3]
arr2 = [[1,1],[2,2],[3,3]]
# 从原数组变成numpy数组
np_arr = np.array(arr, dtype=np.float64)
np_arr2 = np.array(arr2, dtype=np.float64)
# np.dot():矩阵乘法
np_arr3 = np.dot(np_arr,arr2)
# 广播机制
num = 5
np_arr4 = num+np_arr3
# 对np.ndarray可以直接进行常规运算，等价于每个数字进行运算
np_arr5 = (np_arr4)**2 # 乘方
def sigmoid(arr):
    return 1/(1+np.exp(-arr))
np_arr6 = sigmoid(np_arr4)
```

反向传播与更新参数的快速计算：

1. 注意y=sigmoid(x)的导数等于y(1-y)
2. 注意grad_w是拿线性层输出的梯度与线性层输入的矩阵乘法再取均值，grad_b直接拿线性层输出的梯度取均值（取均值都因为损失函数的定义是$\frac{1}{n}×\sum_{i=1}^n(\hat{y_i}-y_i)^2$）

```python
# Backward pass
# Gradient of MSE w.r.t. prediction
dL_dp = 2 * (predictions - labels)                   # shape: (n_samples,)
dp_dz = predictions * (1 - predictions)              # shape: (n_samples,)
dL_dz = dL_dp * dp_dz                                # shape: (n_samples,)

# Gradients
grad_w = np.dot(dL_dz, features) / len(features)     # shape: (n_features,)
grad_b = np.mean(dL_dz)

# Parameter update
weights -= learning_rate * grad_w
bias -= learning_rate * grad_b
```



#### 2025.7.29

- 26. Basic Autograd

```python
class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		# self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		 # Implement addition here
		return Value(self.data+other.data, [self, other], '+')

	def __mul__(self, other):
		# Implement multiplication here
		return Value(self.data*other.data, [self, other], '*')

	def relu(self):
		# Implement ReLU here
		if self.data >=0:
			return Value(self.data, [self], 'relu')
		else:
			return Value(0, [self], 'relu')
	def _backward(self):
		if self._op == 'relu':
			children = [num for num in self._prev]
			if self.data == 0:
				children[0].grad += 0
				children[0]._backward()
			else:
				children[0].grad += self.grad
				children[0]._backward()
		elif self._op == '*':
			children = [num for num in self._prev]
			children[0].grad += children[1].data * self.grad
			children[1].grad += children[0].data * self.grad
			children[0]._backward()
			children[1]._backward()
		elif self._op == '+':
			children = [num for num in self._prev]
			children[0].grad += self.grad
			children[1].grad += self.grad
			children[0]._backward()
			children[1]._backward()
	def backward(self):
		self.grad = 1
		self._backward()
```

- key points:

复现最基础的反向传播梯度更新过程，不涉及矩阵，只有标量

核心思路是从大到小递推解决问题。最大的式子（即调用.backward()的对象）grad为1，然后拆解为子式子的梯度，注意两点：

1. 梯度是累加的，因为一个式子可能有多处需要用到同一变量
2. 子式子的梯度应当在基础上乘以父亲的grad（比如d = a(b+c)，算b的梯度时需要乘以(b+c)这一个父亲式子的梯度a）

