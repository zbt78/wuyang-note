大家好,今天我将要介绍推荐系统。

首先介绍一下推荐系统的分类，主要有基于协调过滤的推荐系统，基于内容的推荐系统，混合推荐系统。其中基于协同过滤的推荐系统包含两类，基于数据统计的cf和基于模型的cf，今天我要着重介绍基于模型的cf——矩阵分解。

在推荐系统中矩阵是长这个样子的，

纵轴是用户，横轴就是这个item，我们要推荐的物品。

矩阵中的数值代表用户有多喜欢这个影视剧，

它可以是个显式的反馈，也可以是隐式的反馈。

显式反馈就是这个观众明确告诉你他喜欢不喜欢，比如说它可以给这个影视剧打分，或者说点赞。

它也可以是一个隐式反馈，就是观众没有提供反馈，但是我们可以从观众的一个互动和行为中推测出他喜不喜欢，比如说他看没看这个视频，他是点击进去之后看了十秒立刻退出，还是说他耐心的去看完了整个视频，一直到最后才退出。

这里我们采取显式反馈作为例子，矩阵中的数值是用户对项目的明确打分，没看或者没打分的就是空白，也就是我们最重要预测的这个目标。

那么问题来了，我们怎么从用户的历史打分记录中来预测用户将来的观看行为呢，也就是说矩阵里面的空白值如何计算呢？

接下来引出用户矩阵和商品矩阵，也就是我们的矩阵分解。

（ppt里有）它指将一个矩阵分解成两个的或多个的矩阵的乘积，通过隐含特征（latent factor）将user兴趣和item特征联系起来。

意思就是我给你一个大大的用户和商品矩阵，把他拆分成两个小矩阵，一个用户矩阵，一个商品矩阵，这个答案肯定是不唯一的，比如说12我可以拆分成3和4，也可以拆分成2和6.最好的答案就是得到了两个小矩阵之后，这两个矩阵相乘后得到的结果要尽可能的接近原来的那个矩阵。



我们可以对求得的用户和物品特征向量来进行一个验证。比如说用户1对物品2的真实评分是5，然后我们使用隐含特征来计算预测的评分，对这两个用户向量和物品向量作内积，-1.0 * -0.8 + 2.3 * 1.8，结果是4.94，与真实的评分5非常接近，说明我们求得的隐向量能够比较准确地表示用户和物品的特征。

然后我们使用隐向量来做一次预测，比如我们想知道第2个用户对第3个物品的评分，首先我们要找到第2个用户和第3个物品对应的隐向量，然后把这两个向量做内积，也就是1.0 * 1.7 + 1.0 *0.5，得到的结果就是为用户2对物品3的评分。

现在目标很明确了，我们模型训练的目标,是输出矩阵和输入矩阵之间的差值最小，输出矩阵的所有单元格都有值,缺失值的填充代表用户评分的预测值。模型训练的输出是用户向量和物品向量,都是K维度,代表K个不同的隐含兴趣点。

然后求解隐向量我们使用梯度下降的方法。思路就是训练用户和物品向量，使得估计的评分来逼近真实评分。

整体来说分两步走：随机初始化用户物品向量 U是用户集合，V是物品集合，Q是物品矩阵，其中d是它的维度，是一个超参数，可以随意定义维度大小，同样地定义一个物品矩阵。

第二步是用梯度下降优化来逼近分解的结果。假设存在真实评分R，那么我们对它估计的评分就是物品和用户向量的内积，这个方程就是梯度下降的损失函数，通过不断训练让矩阵Q和P达到一个理想的状态，进而能估计出一个比较准的真实评分。

在实际使用中我们还会加入正则化的技术，在方程式后面加上Q和P的2范数。假如我们不对这些参数进行限制的话，我们求到的结果只是逼近训练数据，而我们希望训练出这样一个模型，就是它能够比较平滑地来拟合这个数据，而不是像前面这个就会非常完美地拟合训练数据，因为真实的数据分布很有可能和训练数据是由差异的。因此采用正则化来让参数减小，达到正则化的目的。这就是全部了谢谢大家。









Hello everyone, today I will introduce recommendation systems.

Firstly, let's talk about the classification of recommendation systems. Recommendation systems can be mainly divided into collaborative filtering-based recommendation systems, content-based recommendation systems, and hybrid recommendation systems. Collaborative filtering-based can further be classified into neighborhood-based CF and model-based CF. Today, I would like to focus on introducing the model-based CF: matrix factorization.

In a recommendation system, we usually use a matrix to represent users' ratings of items such as movies or music. The vertical axis represents users, while the horizontal axis represents items to be recommended. The numbers in the matrix indicate the degree to which the user likes or dislikes the item. Feedback can be explicit, such as when a user rates a movie, or implicit, such as when a user watches a video for a certain amount of time before exiting or whether a user clicked the item. In this case, we will take explicit feedback as an example, in other words, the values in the matrix are the user's clear ratings of the item, and blank spaces represent no rating or not yet watched. These blank spaces are exactly what we need to predict. If it is implicit feedback, it looks like this.

So, how do we predict which movies a user might like in the future based on their historical rating information? In other words, how do we predict the values in these blank spaces?



Next, we will introduce the user matrix and item matrix, which is matrix factorization that we want to focus on today.

Matrix factorization refers to the process of decomposing a matrix into the product of two smaller matrices or multiple matrices, connecting user interest with item features through latent factors.

In other words, given a large matrix of users and items, it is split into two smaller matrices- a user matrix and an item matrix. The answer is not unique; for example, 12 can be split into 3 and 4 or into 2 and 6. The best solution is to get two smaller matrices so that their product is as close as possible to the original matrix.



We can validate the derived user and item feature vectors, such as User 1 and Item 2. If their true rating in the original matrix was 5, we can use latent features to compute the predicted rating. By taking the dot product of these two vector representations (-1.0 * -0.8 + 2.3 * 1.8), we obtain the value of 4.94 which is reasonably close to 5. This result suggests that our learned latent vectors can accurately represent the features of users and items.

Next, we can make a prediction using these learned latent vectors. For instance, if we want to predict the rating that User 2 would give to Item 3, we need to locate the corresponding latent vectors of User 2 and Item 3, and then take their dot product (1.0 * 1.7 + 1.0 * 0.5) to yield User 2's predicted rating for Item 3.

In summary, our validation demonstrates that the learned latent feature vectors are effective in representing user and item attributes. Utilizing these vectors, we can make accurate predictions on ratings for different items from any given user.

So the target is clear. The training objective of the model is to minimize the difference between the output matrix and the input matrix. All cells in the output matrix have values, with missing values representing predicted ratings. The output of the model is a user vector and an item vector, both having K dimensions, representing K different latent interests.



To find the latent vectors, we use the gradient descent method to train user and item vectors, so that the estimated ratings approach the true ratings.

In general, there are two steps: first, randomly initialize the user and item vectors U for the user set and V for the item set. Define a hyperparameter d as the dimension of the vectors and define a Q matrix for the items.

Secondly, use gradient descent optimization to approximate the decomposition result. Assume that there exist true ratings R, then the estimated rating is the dot product of the item and user vectors. This equation is the loss function for gradient descent. Through continuous training, the Q and P matrices can reach an ideal state, and a more accurate true rating can be estimated.

In actual use, regularization techniques are also added, adding the 2-norm of Q and P after the equation. If we do not limit these parameters, the results we obtain will only approximate the training data. We hope to train a model that can smoothly fit the data, instead of perfectly fitting the training data, because the true data distribution may be different from the training data. Therefore, regularization is used to reduce the parameters and achieve regularization purposes. That's all, thank you.
