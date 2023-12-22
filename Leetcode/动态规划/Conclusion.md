总结dp数组的各种含义

## 一维数组

`dp[i]`表示以第 `i` 项结束的状态。

后续的状态会用到之前的值。

比如说：

- [343. 整数拆分](https://leetcode.cn/problems/integer-break/),dp[i] = max(dp[i], max(j\*i-j, j\*dp[i-j])),先拆成两个数，其中一个固定，看另一个用不用继续拆，是不拆的时候大(j\*i-j)，还是拆的时候大(j\*dp[i-j])
- [96. 不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/),同样的，n个节点可以拆分乘左右两部分，左右两个子部分的节点数肯定比n小，所以在此之前已经计算过了。

## 二维数组



### 2*n

买卖股票问题：

`dp[0]`表示当前天数手里没有股票最大获利

`dp[1]`表示当前手里有一支股票的最大获利



### n*n





