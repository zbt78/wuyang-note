## torch.nn.model类简介
```python
class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
'''
有一部分没有完全列出来
'''
```

我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：
- 一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以把不具有参数的层也放在里面；
- 一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替  
- forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。