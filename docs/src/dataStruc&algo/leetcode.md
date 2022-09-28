## 经典系列问题

### 背包问题

### 股票问题

股票问题是学习动态规划很好的系列，因此一下最先考虑的都是用动态规划怎么做。

【输入】`prices` 表示每一天的股票的价格，长度为 `n`，根据不同题目会有额外的参数，包括最多交易 `k` 次，冷冻期，手续费 `fee`。

【输出】能够获得的最大利润。

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)【简单题】：只能交易一次。矩阵 `dp[n][2]` 表示第 $i$ 天交易完后的最大利润，其中 `dp[i][0]` 表示未持有股票，`dp[i][1]` 表示持有股票。

$dp[i][0]=\max\{dp[i-1][0], dp[i-1][1]+prices[i]\},$

$dp[i][1]=\max\{dp[i-1][1], -prices[i]\}.$

其中边界条件：`dp[0][0] = 0`，`dp[0][1] = -prices[0]`. 

实际上，`dp[i][1]` 就是在记录过去的天数中最低的价格。这个问题最简单的思路，其实就是记录以往的最低价格，然后用今天的价格出售，动态维护最大值即为答案。

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)【中等题】：能够交易无限次。同样的，矩阵 `dp[n][2]` 表示第 $i$ 天交易完后的最大利润，其中 `dp[i][0]` 表示未持有股票，`dp[i][1]` 表示持有股票。

$dp[i][0]=\max\{dp[i-1][0], dp[i-1][1]+prices[i]\},$

$dp[i][1]=\max\{dp[i-1][1], dp[i-1][0]-prices[i]\}.$

其中边界条件：`dp[0][0] = 0`，`dp[0][1] = -prices[0]`. 最终答案为：`dp[n - 1][0]`. 

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)【中等题】：每笔交易都要支付手续费 `fee`. 同样的方式，只是在卖出的时候要多计算一个手续费

$dp[i][0] = \max\{dp[i - 1][0], dp[i - 1][1] + prices[i] - fee\},$

$dp[i][1] = \max\{dp[i-1][1],dp[i-1][0]-prices[i]\}.$

其中边界条件：`dp[0][0] = 0`，`dp[0][1] = -prices[0]`. 最终答案为：`dp[n - 1][0]`. 

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)【困难题】：最多完成两笔交易。定义 4 个状态，分别是：`buy1` 进行一次买入、`sell1` 进行一次卖出、`buy2` 进行第二次买入、`sell2` 进行第二次卖出。

$buy_1 = \max\{buy_1, -prices[i]\},$

$sell_1=\max\{sell_1, buy_1+prices[i]\},$

$buy_2 = \max{buy_2, sell_1-prices[i]},$

$sell_2=\max\{sell_2,buy_2+prices[i]\}.$

其中边界条件：`buy1 = -prices[0]`，`sell1 = 0`，`buy2 = -prices[0]`，`sell2 = 0`，答案就是 `sell2`. 

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)【困难题】：最多只能够完成 `k` 笔交易。定义矩阵 `buy[n][k + 1]` 和 `sell[n][k + 1]` 分别表示第 $i$ 天时完成第 $j$ 笔交易后持有以及未持有股票的最大利润。

$buy[i][j] = \max\{buy[i-1][j],sell[i-1][j]-prices[i]\},$

$sell[i][j]=\max\{sell[i-1][j],buy[i-1][j-1]+prices[i]\}.$

其中边界条件：`buy[0][0] = -prices[i]`，`buy[0][1...k] = float('-inf')`，`sell[0][0] = 0`，`sell[0][1...k] = float('-inf')`。`j = 0` 时，无需对 `sell[i][j]` 进行处理，保持为 0。

最终答案为 `sell[n - 1][0...k]` 中的最大值。

> 注意：因为 `n` 天最多只能进行 $\lfloor\frac{n}{2}\rfloor$ 次交易，因此在一开始时可以 `k = min(k, n // 2)`。

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)【中等题】：冷冻期为 1 天（卖出股票后得隔一天才能买入股票）。矩阵 `dp[n][3]` 存储第 $i$ 天结束时的最大收益，其中 `dp[i][0]` 表示持有一支股票，`dp[i][1]` 表示不持有任何股票并且处于冷冻期，`dp[i][2]` 表示不持有股票并且不处于冷冻期。

$dp[i][0] = \max\{dp[i-1][0], dp[i-1][2] - prices[i]\},$

$dp[i][1] = dp[i-1][0] + prices[i],$

$dp[i][2]=\max\{dp[i-1][1],dp[i-1][2]\}.$

其中边界条件：`dp[0][0] = -prices[0]`，`dp[0][1] = 0`， `dp[0][2] = 0`. 最终答案为 `max(dp[n - 1][1], dp[n - 1][2])`. 

### 岛屿问题

