## pytorch模型构建

PyTorch 构建模型需要 5 大步骤：

-   数据：包括数据读取，数据清洗，进行数据划分和数据预处理，比如读取图片如何预处理及数据增强。
	
-   模型：包括构建模型模块，组织复杂网络，初始化网络参数，定义网络层。
	
-   损失函数：包括创建损失函数，设置损失函数超参数，根据不同任务选择合适的损失函数。
	
-   优化器：包括根据梯度使用某种优化器更新参数，管理模型参数，管理多个参数组实现不同学习率，调整学习率。
	
-   迭代训练：组织上面 4 个模块进行反复训练。包括观察训练效果，绘制 Loss/ Accuracy 曲线，用 TensorBoard 进行可视化分析。

```py
#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

#%%
# ============================ step 1/5 生成数据 ============================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
# 使用正态分布随机生成样本，均值为张量，方差为标量
x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
# 生成对应标签
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1)
# 使用正态分布随机生成样本，均值为张量，方差为标量
x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
# 生成对应标签
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

#%%
# ============================ step 2/5 选择模型 ============================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        # 输入特征数为2，输出特征数为1
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()   # 实例化逻辑回归模型

#%%
# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()

#%%
# ============================ step 4/5 选择优化器   ============================
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

#%%
# ============================ step 5/5 模型训练 ============================
for iteration in range(1000):

    # 前向传播
    y_pred = lr_net(train_x)
    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad()
    # 绘图
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()
        # plt.savefig(str(iteration / 20)+".png")
        plt.show()
        plt.pause(0.5)
        # 如果准确率大于 99%，则停止训练
        if acc > 0.99:
            break
# %%

```



## 数据加载

数据模块可以细分为 4 个部分：
-   数据收集：样本和标签。
    
-   数据划分：训练集、验证集和测试集
    
-   数据读取：对应于PyTorch 的 DataLoader。其中 DataLoader 包括 Sampler 和 DataSet。Sampler 的功能是生成索引， DataSet 是根据生成的索引读取样本以及标签。
    
-   数据预处理：对应于 PyTorch 的 transforms

![img](https://blog-img-zbt.oss-cn-beijing.aliyuncs.com/picture/wuyang/202301302247911.png)

### DataLoader 与 DataSet

#### torch.utils.data.DataLoader()
```python
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)
```

功能：构建可迭代的数据装载器

- dataset: Dataset 类，决定数据从哪里读取以及如何读取
	
- batchsize: 批大小
	
- num_works:num_works: 是否多进程读取数据
	
- shuffle: 每个 epoch 是否乱序
	
- drop_last: 当样本数不能被 batchsize 整除时，是否舍弃最后一批数据

#### Epoch, Iteration, Batchsize

-   Epoch: 所有训练样本都已经输入到模型中，称为一个 Epoch
    
-   Iteration: 一批样本输入到模型中，称为一个 Iteration
    
-   Batchsize: 批大小，决定一个 iteration 有多少样本，也决定了一个 Epoch 有多少个 Iteration

#### torch.utils.data.Dataset

功能：Dataset 是抽象类，所有自定义的 Dataset 都需要继承该类，并且重写`__getitem()__`方法和`__len__()`方法 。`__getitem()__`方法的作用是接收一个索引，返回索引对应的样本和标签，这是我们自己需要实现的逻辑。`__len__()`方法是返回所有样本的数量。

数据读取包含 3 个方面

-   读取哪些数据：每个 Iteration 读取一个 Batchsize 大小的数据，每个 Iteration 应该读取哪些数据。
    
-   从哪里读取数据：如何找到硬盘中的数据，应该在哪里设置文件路径参数
    
-   如何读取数据：不同的文件需要使用不同的读取方法和库。

## GPU加速

```python
# 将网络模型在gpu上训练
model = Model()
model = model.cuda()

# 损失函数在gpu上训练
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 数据在gpu上训练
for data in dataloader:                        
	imgs, targets = data
	imgs = imgs.cuda()
	targets = targets.cuda()
```