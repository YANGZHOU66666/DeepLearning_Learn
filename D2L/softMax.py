import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
'''
# print(len(mnist_test),len(mnist_train))
# print(mnist_train[0][0].shape)
'''

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
'''
# 定义一个函数来展示图片
def show_images(images, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image.squeeze().numpy())
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    return axes

# 选择一小部分图片进行展示
sample_images = [mnist_train[i][0] for i in range(10)]
sample_labels = [mnist_train[i][1] for i in range(10)]
show_images(sample_images, 2, 5, sample_labels)
plt.show()
'''
'''
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据,但似乎不能用于windows系统"""
    return 4
'''

'''
timer = d2l.Timer()
for X, y in train_iter:
    print(X.shape,y.shape)
    continue
print(f'{timer.stop():.2f} sec')
'''

batch_size = 16
lr = 0.01
num_epochs = 1
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=0)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0)

num_inputs=28*28
num_outputs=10

W = torch.normal(0,1,(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b) # reshape(-1,W.shape[0]) 指将X变为一维

def cross_entropy(y_hat, y): # 交叉熵损失函数
    return -torch.log(y_hat[range(len(y_hat)), y]) # 相当于只取y_hat中和y等于1的那一维度

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


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


def evaluate_accuracy(data_iter, net):
    if isinstance(net, torch.nn.Module): # 如果传入的net是pytorch库里定义好的
        net.eval() # 评估模式, 这会关闭dropout
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad(): # 关闭自动求导
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # y.numel() 返回y的元素个数(numel:num of elements)
    return metric[0] / metric[1]
# print(evaluate_accuracy(train_iter, net))

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

def updater(batch_size): # 定义自定义的优化器
    return d2l.sgd([W, b], lr, batch_size)


# 检查是否有可用的GPU
is_gpu_available = torch.cuda.is_available()
print(is_gpu_available)

# 检查当前默认的GPU编号
current_device = torch.cuda.current_device()
print(current_device)

train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_image(img):
    X = img.reshape((-1,W.shape[0]))
    y_hat = net(X)
    return get_fashion_mnist_labels(y_hat.argmax(dim=1).item())

imgs = []
labels = []
for X, y in test_iter:
    imgs=X.reshape((-1,W.shape[0]))
    labels=y
    break

for i in range(10):
    print(get_fashion_mnist_labels(net(imgs[i]).argmax(dim=1)),get_fashion_mnist_labels([labels[i]]))