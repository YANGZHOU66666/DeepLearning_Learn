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

