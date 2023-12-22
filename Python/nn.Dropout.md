1. 1.Dropout是为了防止过拟合而设置的
2. Dropout顾名思义有丢掉的意思
3. nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活
4. Dropout只能用在训练部分而不能用在测试部分
5. Dropout一般用在全连接神经网络映射层之后，如代码的nn.Linear(20, 30)之后

原理是以一定的随机概率将卷积神经网络模型的部分参数归零，以达到减少相邻两层神经连接的目的