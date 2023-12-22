![[Pasted image 20221021160457.png]]
**Precision**: 
$$
Precision = {True\;positive \over predicted\;positive} = {True\;positive \over True\;positive\;+\;False\;positive}
$$
**Recall** = True positive / actual positive = True positive / (True positive + False negative)

**F1 socre:**
![[Pasted image 20221021162410.png]]

**CG**: cumulative gain 累计增益 
**DCG**: Discounted CG 折损累计增益
$$
DCG@K\,\,=\,\,\sum_{i=1}^k{\frac{r\mathrm{e}l_i}{\log _2\left( i+1 \right)}}
$$


**IDCG**: 理想情况下最大的DCG值
$$
IDCG@K\,\,=\,\,\sum_{i=1}^{\left| REL \right|}{\frac{r\mathrm{e}l_i}{\log _2\left( i+1 \right)}}
$$

**NDCG**: Normalized DCG 归一化折损累计增益
$$
NDCG = {DCG \over IDCG}
$$
这三项全是基于排名的，分母$log_2{(i+1)}$相当于对返回位置进行了惩罚，越往后惩罚越大.


**AUC**: 评估排名的表现
$$
AUC\,\,=\,\,\frac{\sum{_{ins_i\in positiv\mathrm{e}class}rank_{ins_i}}-\frac{M\times \left( M+1 \right)}{2}}{M\times N}
$$
M为正样本数量，N为负样本数量


![[Pasted image 20221108181654.png]]
![[Pasted image 20221108181711.png]]
AUC优势，AUC的计算方法同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。例如在反欺诈场景，设欺诈类样本为正例，正例占比很少（假设0.1%），如果使用准确率评估，把所有的样本预测为负例，便可以获得**99.9%的准确率**。


**NLL**：评估预测的表现

