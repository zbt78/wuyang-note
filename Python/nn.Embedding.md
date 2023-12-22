```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
```

参数：
- num_embeddings：词典的大小尺寸，比如总共出现5000个词，那就输入5000
- embedding_dim：嵌入向量的维度，即用多少维来表示一个符号

变量：
Embedding.weight –形状模块（num_embeddings，Embedding_dim）的可学习权重，初始化自(0,1)。**也就是说，pytorch的nn.Embedding()是可以自动学习每个词向量对应的w权重的**

[公主王妃的例子](https://www.cnblogs.com/USTC-ZCC/p/11068791.html)
