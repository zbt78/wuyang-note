在本文中，解决问题的一个方式是每次优化一个item对，优化的并不是用户对单个item的评分，而是用户对两个item的偏好关系。

$\left( u,i,j \right) \in D_S$ 就表示 $u$ 在 $i,j$ 中更喜欢 $i$ ，此为正样本，那么反过来 $(u,j,i)$ 即为负样本。
![[Pasted image 20221103160324.png]]
这样处理的**优点**：处理后data既包含正样本也包含负样本以及缺失信息的样本。比如两个都没有交互的物品之间的序关系就是未知的，是我们要学的；

## Pytorch实现

实现 Bayesian Personalized Ranking (BPR) 模型在 PyTorch 中可以使用以下步骤：

1.  定义模型结构，这通常是一个矩阵分解模型。在 PyTorch 中可以使用 nn.Embedding 层来学习用户和物品的嵌入向量。
2.  定义损失函数。BPR 模型使用的损失函数是负对数似然，可以使用 PyTorch 提供的 BPRLoss 损失函数。
3.  训练模型。通过输入用户-物品对和标签 (1 或 -1) 训练模型。
4.  预测。在训练完成后，可以使用训练得到的用户和物品嵌入向量来预测用户对物品的评分。

下面是一个示例代码：
```python
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

class BPR(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(BPR, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, emb_size)
        self.item_embeddings = nn.Embedding(num_items, emb_size)

    def forward(self, user_id, item_id):
        user_emb = self.user_embeddings(user_id)
        item_emb = self.item_embeddings(item_id)
        return (user_emb * item_emb).sum(dim=1)

class BPRLoss(nn.Module):
    def forward(self, x):
        return -torch.log(sigmoid(x))

class BPRDataset(Dataset):
    def __init__(self, user_item_pairs, labels):
        self.user_item_pairs = user_item_pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
    return (self.user_item_pairs[idx][0], self.user_item_pairs[idx][1]), self.labels[idx]

```


```python
# 定义超参数：
num_users = 100 
num_items = 200 
emb_size = 20 
num_epochs = 10 
batch_size = 512

# 定义模型:
model = BPR(num_users, num_items, emb_size)

# 定义损失函数:
loss_fn = BPRLoss()

# 定义优化器:
optimizer = Adam(model.parameters())

# 加载数据
user_item_pairs, labels = load_data() 
data_set = BPRDataset(user_item_pairs, labels) 
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs): 
	for user_ids, item_ids, labels in data_loader: 
		optimizer.zero_grad() 
		scores = model(user_ids, item_ids) 
		loss = loss_fn(scores * labels) 
		loss.backward() 
		clip_grad_norm_(model.parameters(), 1.0) 
		optimizer.step()

# 保存模型
torch.save(model.state_dict(), 'bpr.pt')

# 预测
user_ids = torch.tensor([1, 2, 3]) 
item_ids = torch.tensor([4, 5, 6]) 
scores = model(user_ids, item_ids)

""" 
上面这段代码中的load_data()函数应该是数据载入的函数，用于载入训练数据。训练数据中应该包括用户-物品对和对应的标签(1或-1)
请注意这仅仅是一个示例代码，还需要根据你自己的数据来修改。
"""
```

上面的代码只是实现 BPR 模型的一种方式，在实际应用中可能需要根据需求进行更改。

例如:

-   可能需要对嵌入向量进行预训练，以提高模型的性能。
-   可能需要使用不同的优化器来训练模型，例如 Adagrad 或 AdamW。
-   可能需要使用不同的损失函数来训练模型，例如 hinge 损失或 logistic 损失。
-   可能需要添加更多的其他特征来提高模型的性能。

在实现 BPR 模型时，最重要的是要理解 BPR 算法的基本原理，并根据实际应用来进行模型调整和优化。

另外,在训练BPR模型之前,你需要做的是处理原始数据, 将其转换成适当的格式，便于输入给模型。 这可能包括：

-   将数据转换成用户-物品对的形式
-   移除缺失值
-   离散化连续特征
-   将用户和物品的编号映射到从0开始的连续整数。

数据预处理是一个相当重要的步骤，因为这直接影响模型的性能。

此外,需要注意的是,对于训练数据中的用户-物品对, 在训练中使用的是正样本和负样本，负样本是在正样本数据上随机采样得到的。 在BPR模型中，我们优化的是排列推荐中正样本排在负样本之前的概率。

在评估 BPR 模型的性能时，常用的评估指标包括：

-   AUC-ROC (Receiver Operating Characteristic - Area Under the Curve)：用于评估模型的二分类性能。AUC-ROC 越大，模型的性能越好。
-   Precision，Recall，F1-Score：这些指标用于评估模型的精确度和召回率。
-   MRR (Mean Reciprocal Rank)：衡量在推荐结果中第一个正样本出现的位置。 MRR 越高，说明推荐更准确。
-   NDCG (Normalized Discounted Cumulative Gain)：衡量推荐结果中前 n 个正样本的相对重要性。
-   Hit Rate：衡量推荐结果中是否包含了某些特定的正样本。

根据具体应用场景不同, 应该会选用不同的评价指标。 例如在绝大部分推荐系统场景下, MRR 或 NDCG是比较常用的评价指标.

另外，需要注意的是，在计算评估指标时，通常需要使用离线评估的方法，即将模型在训练数据上训练好后，再使用测试数据来评估模型的性能。

pytorch实现AUC：
```python
from sklearn.metrics import roc_auc_score

# 加载测试数据
test_user_item_pairs, test_labels = load_test_data()

# 使用测试数据预测模型的评分
model.eval()
with torch.no_grad():
    test_scores = model(test_user_item_pairs[:, 0], test_user_item_pairs[:, 1])
test_scores = test_scores.detach().cpu().numpy()

# 计算 AUC
auc = roc_auc_score(test_labels, test_scores)
print('AUC:', auc)

"""
其中`load_test_data`函数用于加载测试数据。 这里我们使用了sklearn库中的roc_auc_score来计算AUC, 它可以对二分类问题计算出曲线下面积（AUC）,然后输出。

在计算 AUC 时，需要将模型的输出作为概率估计值，这可以通过在模型输出上应用 sigmoid 函数来实现。 请注意，在这里, 我使用了no_grad来禁用PyTorch自动求导，因为在评估指标时不需要进行反向传播。
"""
```