### Inroduction
Bias:
- the exposure policy of a recommender system(exposure bias)
- other users' actions
- popular items have more chances to be clicked(popular bias)
- the demographic, spatial and temporal heterogeneity in population induces the shift of user or item distributions

1. 使用 random logging policy 来采集 uniform data，然后来监督model在有偏数据集上的训练
2. 要找到由不能观察的干扰因子造成的潜在的偏差，然后得到普遍的去偏
3. 通过估计**伪环境标签**作为**代理**来捕获为观察到的干扰因子
4. 通过对抗学习的方式将不变偏好（即环境无关偏好）和变体偏好（即环境相关偏好）从可以观察的行为区分开来
5. **在所有环境中找到y_hat最接近y的环境**
6. Embedding就是用一个低维的向量表示一个物体，这个Embedding向量的性质是能使距离相近的向量对应的物体有相近的含义



