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

$buy_2 = \max\{buy_2, sell_1-prices[i]\},$

$sell_2=\max\{sell_2,buy_2+prices[i]\}.$

其中边界条件：`buy1 = -prices[0]`，`sell1 = 0`，`buy2 = -prices[0]`，`sell2 = 0`，答案就是 `sell2`. 

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)【困难题】：最多只能够完成 `k` 笔交易。定义矩阵 `buy[n][k]` 和 `sell[n][k]` 分别表示第 $i$ 天时处于第 $j$ 笔交易时持有以及未持有股票的最大利润。

$buy[i][j] = \max\{buy[i-1][j],sell[i-1][j-1]-prices[i]\},$

$sell[i][j]=\max\{sell[i-1][j],buy[i-1][j]+prices[i]\}.$

其中边界条件：`buy[0][0...k] = -prices[i]`，`sell[0][0...k] = 0`。当 `j = 0` 时，`buy[i][j] = max(buy[i - 1][j], sell[i - 1][j - 1] - prices[i])`. 

最终答案为 `sell[n - 1][0...k]` 中的最大值。

> 注意：因为 `n` 天最多只能进行 $\lfloor\frac{n}{2}\rfloor$ 次交易，因此在一开始时可以令 `k = min(k, n // 2)`，如果 `k == 0` 那么答案直接为 0. 

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)【中等题】：冷冻期为 1 天（卖出股票后得隔一天才能买入股票）。矩阵 `dp[n][3]` 存储第 $i$ 天结束时的最大收益，其中 `dp[i][0]` 表示持有一支股票，`dp[i][1]` 表示不持有任何股票并且处于冷冻期，`dp[i][2]` 表示不持有股票并且不处于冷冻期。

$dp[i][0] = \max\{dp[i-1][0], dp[i-1][2] - prices[i]\},$

$dp[i][1] = dp[i-1][0] + prices[i],$

$dp[i][2]=\max\{dp[i-1][1],dp[i-1][2]\}.$

其中边界条件：`dp[0][0] = -prices[0]`，`dp[0][1] = 0`， `dp[0][2] = 0`. 最终答案为 `max(dp[n - 1][1], dp[n - 1][2])`. 

### 岛屿问题

岛屿问题是学习深度优先搜索（DFS）和广度优先搜索（BFS）的经典系列问题。

【输入】`grid` 是一个 $M$ 行 $N$ 列的矩阵，其中 1 代表陆地，0 代表海洋。

【输出】不同的问题有不同的要求，例如岛屿数量、岛屿周长、岛屿面积等

[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)【简单题】：计算岛屿数量。

DFS：扫描网格，碰到某个位置为 1，以其为起始节点开始进行深度优先搜索。将每个碰到的 1 都变成 0，避免重复搜索。最终的答案就是进行 DFS 的次数。

BFS：扫描网格，如果某个位置为 1，将其加入队列，开始 BFS，没搜索到的 1 都重新标记为 0，直到队列为空。最终的答案就是进行 BFS 的次数。

[305. 岛屿数量 II](https://leetcode.cn/problems/number-of-islands-ii/)【困难题】：最初全部为海洋，`positions[i] = [ri, ci]` 记录的是每次添加的陆地位置，要求返回 `answer` 记录每次将 `[ri, ci]` 变成陆地后地图中的岛屿数量。

> 这个问题在题解里只找到并查集的解法。

并查集：

[463. 岛屿的周长](https://leetcode.cn/problems/island-perimeter/)【简单题】：`grid` 中只有一个岛屿，且岛屿中没有湖，要求返回这个岛屿的周长。

DFS：扫描网格，碰到第一个为 1 的位置，开始搜索，如果碰到海洋或者边界，则周长加 1，否则继续搜索。通过将 `grid[x][y]` 添加到集合中或者重新标记为 2 来避免重复搜索。

BFS：扫描网格，碰到第一个为 1 的位置，将其加入队列并开始搜索，如果碰到海洋或者边界，则周长加 1，搜索到的位置添加到已遍历集合中或者标记为 2，直到队列为空。

[694. 不同岛屿的数量](https://leetcode.cn/problems/number-of-distinct-islands/)【中等题】：返回 `grid` 中有多少种不同形状的岛屿。相同形状的岛屿指的是可以通过平移与另一个岛屿重合（不可旋转、翻转）。

[695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)【中等题】：返回 `grid` 中最大岛屿的面积。

DFS：扫描网格，碰到某个位置为 1，开始 DFS，将每个碰到的 1 都重新标志为 0，并且面积加 1，遍历完之后动态维护全局最大面积。

BFS：扫描网格，碰到某个位置为 1，加入队列，开始 BFS，将每个碰到的 1 都重新标志为 0，并且面积加 1，直到队列为空。

序列化遍历顺序：用 `[1, 2, 3, 4]` 分别代表上下左右四个方向的进口，出口则分别对应 `[-1, -2, -3, -4]`，这样遍历完一个岛屿，就能够得到一个序列，将这个序列保存到一个集合中。最终的答案就是集合中不同种序列的数量。

[1254. 统计封闭岛屿的数目](https://leetcode.cn/problems/number-of-closed-islands/)【中等题】：定义「岛」是由最大的 4 个方向连通的陆地组成的群；「封闭岛」是一个 完全由海洋包围的岛。题目要求返回封闭岛屿的数量。

> 这道题目中陆地和海洋的标志互换了，这里还是依照 1 代表陆地，0 代表海洋来讨论。

BFS：如果 BFS 中碰到了边界，则不是一个封闭岛。碰到某个位置为 1，加入队列开始 BFS，将碰到的每个 1 都重新标志为 0，如果碰到了边界，则记录该陆地不是封闭岛，不断重复直到队列为空。如果该陆地为封闭岛则数量加 1.

[1905. 统计子岛屿](https://leetcode.cn/problems/count-sub-islands/)【中等】：有两个同样大小的 `grid1` 和 `grid2`，如果 `grid2` 中的岛屿中每个位置在 `grid1` 中的对应位置都是陆地，那么这个岛屿就称为「子岛屿」，题目要求返回 `grid2` 中子岛屿的数目。

DFS：扫描网格 `grid2`，碰到某个位置为 1，开始搜索，每个 1 都判断在 `grid1` 中是否也为 1，如果不是则记录该岛不是一个子岛屿，然后将遇到的每一个 1 都重新标志为 0. 在遍历完 `grid2` 中的一个岛之后，如果是一个子岛屿，则将子岛屿的数量加 1. 

BFS：扫描网格 `grid2`，碰到某个位置为 1，加入队列开始搜索，每个 1 都判断在 `grid1` 中是否也为 1，如果不是则记录该岛不是一个子岛屿，然后将遇到的每一个 1 都重新标志为 0. 在遍历完 `grid2` 中的一个岛之后，如果是一个子岛屿，则将子岛屿的数量加 1. 



### 排列、组合、子集问题

排列、组合、子集问题是学习回溯算法的一系列很好的问题。回溯算法就是在一棵树上的深度优先遍历（DFS）。

[46. 全排列](https://leetcode.cn/problems/permutations/)【中等】

【输入】一个不包含重复数字的序列 `nums`

【输出】返回所有不重复的全排列（不考虑顺序）

递归函数 `dfs(nums, temp, used)`，`temp` 是当前已经决策的路径，`used` 记录对应数字是否已使用：

- 如果 `len(temp) == len(nums)`，将 `temp` 放入答案数组中，找到其中一个答案；
- 定义一个标记数组 `used`，递归时遍历 `nums` 找到没被选择的数字填入 `temp`，继续递归；
- 回溯的时候要撤销 `temp` 中的数的数以及标志，继续尝试其他没被标记过的数；

[47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)【中等】

【输入】一个包含重复数字的序列 `nums`

【输出】返回所有不重复的全排列（不考虑顺序）

递归函数 `dfs(nums, temp, used)`，`temp` 是当前已经决策的路径，`used` 记录对应数字是否已使用：

- 首先对数组 `nums` 排序，再开始递归；
- 如果 `len(temp) == len(nums)`，将 `temp` 放入答案数组中，找到其中一个答案；
- 定义一个标记数组 `used`，递归时遍历 `nums` 找到没被选择的数字填入 `temp`，继续递归；
  - 避免重复：将 `nums` 进行排序之后，相同的数字会在一起，如果当前数字与前一个数字相同，并且前一个数字没有被使用 `i > 0 and nums[i] == nums[i - 1] and not used[i - 1]`，则跳过当前的数字；

- 回溯的时候要撤销 `temp` 中的数以及标志，继续尝试其他没被标记过的数；

> 注意：组合与排列的不同，排列问题是关注顺序的，而组合问题不关注。具体而言，对于数组 `[1, 2, 3]` 和 `[1, 3, 2]` 在组合问题中是一样的，而在排列问题中则不一样。

[39. 组合总和](https://leetcode.cn/problems/combination-sum/)【中等】

【输入】一个无重复元素的整数数组 `candidates` 和目标整数 `target`

【输出】返回 `candidates` 中可以使得数字和为 `target` 的所有不同组合所组成的列表（不考虑顺序），同一个数字可以重复使用

递归函数 `dfs(candidates, target, combine, begin)` 表示还剩 `target` 要组合，已经组合的列表为 `combine`，从 `begin` 位置去遍历：

- 剪枝：首先对 `candidates` 进行排序，然后开始递归；

- 当 `target == 0` 时将当前组合 `combine` 加入到答案列表中；

- 遍历 `candidates` 的 `i = [begin, ..., n]` 位置：
  - 当 `target - candidates[i] < 0` 则停止遍历【剪枝】；
  - 否则将当前数字加入 `combine` 中，将当前位置 `i` 作为 `begin` （因为数字可以重复使用）继续递归 `dfs(candidates, target - candidates[i], combine, i)`；


[40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)【中等】

【输入】一个含重复元素的整数数组 `candidates` 和目标整数 `target`

【输出】返回 `candidates` 中使得数字和为 `target` 的所有不同组合所组成的列表（不考虑顺序），同一个数字只能使用一次

递归函数 `dfs(candidates, target, combine, begin)`，表示还剩 `target` 要组合，已经组合的列表为 `combine`，从 `begin` 位置去遍历：

- 剪枝：首先对 `candidates` 进行排序，然后开始递归；

- 当 `target == 0` 时将当前组合 `combine` 加入到答案列表中；
- 遍历 `candidates` 的 `i = [begin, ..., n]` 位置：
  - 当 `target - candidates[i] < 0` 则停止遍历【剪枝】；
  - 如果 `i > begin and candidates[i - 1] == candidates[i]` 则跳过当前的数；
  - 否则将当前数字加入 `combine` 中，将下一个位置 `i + 1` 作为 `begin` （因为数字不可以重复使用）继续递归 `dfs(candidates, target - candidates[i], combine, i + 1)`；

[77. 组合](https://leetcode.cn/problems/combinations/)【中等】

【输入】两个整数 `n` 和 `k`

【输出】返回范围 `[1, n]` 中所有的可能的 `k` 个数的组合

递归函数 `dfs(cur, n, k, temp)` 表示当前在位置 `cur`，数组范围为 `n` 要选择 `k` 个数字进行组合，`temp` 数组用于记录当前已经选择的数字：

- 如果 `temp`  的长度已经等于 `k`，那么记录当前的组合；
- 对区间 `i in [cur, ..., n]` 进行遍历：
  - 剪枝：如果 `temp` 的长度加上区间 `[cur, n]` 的长度小于 `k`，那么不可能构造出长度为 `k` 的 `temp`，直接结束遍历；
  - 考虑当前位置，则 `temp.append(i)`；
  - 不考虑当前位置，则直接对下一个位置进行决策 `dfs(i + 1, n, k, temp)`；

[78. 子集](https://leetcode.cn/problems/subsets/)【中等】

【输入】一个不含重复元素的整数数组 `nums`

【输出】返回该数组所有可能的子集（不考虑顺序）

递归函数 `dfs(nums, idx, temp)` 代表当前决策数组 `nums` 中的第 `idx` 个位置，临时数组 `temp` 中为决定放进子集的数字：

- 如果当前已经决策完所有的数字 `idx == len(nums)`，则把 `temp` 作为答案放进答案数组 `self.ans.append(temp.copy())`；
- 考虑添加当前位置的数字：将当前位置加入临时数组 `temp`，则 `temp.append(nums[cur])`，再对下一个位置进行决策；
- 考虑不添加当前位置的数字：直接对下一个位置进行决策 `dfs(nums, idx + 1, temp)`；

[90. 子集 II](https://leetcode.cn/problems/subsets-ii/)【中等】

【输入】包含重复元素的整数数组 `nums`

【输出】返回该数组所有不重复的子集（不考虑顺序）

避免重复的方法：先对数组 `nums` 进行排序，保证相同的数字都在一起。如果没有选择上一个数，并且当前数字与上一个数相同，则跳过当前生成的子集。

递归函数 `dfs(nums, idx, temp, used)` 代表当前决策数组 `nums` 中的第 `idx` 个位置，临时数组 `temp` 中为决定放进子集的数字，`used` 有哪些数字被选择：

- 先对 `nums` 进行排序，再开始递归；
- 如果 `idx == len(nums)`，则将数组加入到答案中；
- 对于当前的位置 `idx`：
  - 首先考虑不添加当前位置的数字 `dfs(nums, idx + 1, temp, used)`；
  - 如果 `cur > 0 and nums[cur - 1] == nums[cur] and not used[idx]`，则直接 `return`；
  - 否则，将当前位置加入临时数组 `temp`，即 `temp.append(nums[idx])`，再对下一个位置进行决策；

> 注意：考虑不添加当前位置的情况要放在判断是否重复前面，否则答案数组中会没有「空集」

[60. 排列序列](https://leetcode.cn/problems/permutation-sequence/)【困难】

【输入】给定 `n` 和 `k`

【输出】返回集合 `[1, 2, 3,..., n]` 所有排列的第 `k` 个

这个题目可以很好地学习剪枝的思想，基本的思路是 [46. 全排列](https://leetcode.cn/problems/permutations/) 再加上剪枝。

剪枝：在每个分支里，根据还未选定的个数计算阶乘 $(n - index)!$ 来得到当前分支有多少个叶子结点：

- 如果 `k` 大于这个分支的叶子结点数，代表答案肯定不在这个分支里，直接跳过当前分支；
- 如果 `k` 小于这个分支的叶子结点数，代表答案肯定在这个分支里，继续递归进行剪枝；

递归函数 `dfs(n, k, index, temp, used)` 表示当前决策第 `index` 个数，`temp` 保存已经决策完的 `index - 1` 个数，`used` 用于标记已经使用过的数字。

- 如果 `index == n` 则代表 `temp` 为要返回的第 `k` 个序列，即为答案；
- 每一次 DFS 都要在 `i = [1, 2,..., n]` 中找到没有使用的数字 `used[i] == False`：
  - 计算 `cnt = factorial[n - index]`；
  - 如果 `cnt < k` 则要对这个分支进行剪枝，令 `k -= cnt` 然后寻找下一个未被使用的数字；
  - 否则 `cnt >= k` 代表答案在当前分支里，选择当前数字 `temp.append(i)`，标记为已使用 `used[i] = Ture`，继续递归决策下一个数 `dfs(n, k, index + 1, temp, used)`；

> 由于题目中 $n\in[1, 9]$，因此可以弄一个数组存放不同数字对应的阶乘：`[1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]`



> 以下未整理

[93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)【中等】

【输入】只包含数字的字符串 `s`

【输出】输出所有可能的有效 IP 地址

> 有效 IP 地址：由四个 `[0, 255]` 之间的整数组成，例如 `0.1.2.201`, `192.168.1.1`, `255.255.255.255`

递归函数 `dfs(segId, segStart)` 表示从 `s[segStart]` 开始，搜索 IP 地址中的第 `segId` 段，其中 $\text{segId}\in\{1,2,3,4\}$：

- 从小到大依次枚举当前这一段 IP 地址的结束位置 `segEnd`，如果满足要求，递归进行下一段 `dfs(segId + 1, segEnd + 1)`；
- 由于不能包含前导零，如果 `s[segStart] == '0'` 那么第 `segId` 段只能为 0；
- 如果已经得到全部 4 段地址，并且遍历完整个字符串 `segId = 4 and segStart = len(s)`，则将其加入答案；

[733. 图像渲染](https://leetcode.cn/problems/flood-fill/)【简单】

【输入】大小为 `(m, n)` 的二维数组 `image`，着色位置 `sr, sc` 和颜色 `newColor`，`image[i][j]` 表示该点的像素值；`sr, sc` 表示从位置 `image[sr][sc]` 开始着色 `newColor`

【输出】返回经过上色渲染之后的图像

> 上色渲染：将所有上下左右颜色相同的位置都变成 `newColor`

递归函数 `dfs(image, x, y, preColor, newColor)`：

- 先记录 `image[sr][sc]` 的颜色 `preColor`，如果 `preColor` 与 `newColor` 一样，那么直接返回，那不用递归；
- 不断地看当前位置的上下左右四个方向，如果 `image[x][y]` 的颜色与 `preColor` 一样，就递归修改；

[130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)【中等】

【输入】大小为 `(m, n)` 的由若干 `X` 和 `O` 组成的二维数组 `board`

【输出】不输出任何内容，直接原地修改（将所有被 `X` 围绕的 `O` 都变成 `X`）

实际上，任何不与边界相连的 `O` 都是被 `X` 围绕的。

递归函数 `dfs(board, x, y)`，每寻找到一个边界的 `O` 就开始递归：

- 将与边界相连的 `O` 都标记为 `A`；
- 递归完成之后，如果 `board[x][y] == 'A'` 代表与边界相连，那么重新标记为 `O`；如果 `board[x][y] == 'O'` 代表没有与边界相连，那么重新标记为 `X`；

[79. 单词搜索](https://leetcode.cn/problems/word-search/)【中等】

【输入】大小为 `(m, n)`  的数组 `board`，一个字符串 `word`

【输出】返回 `word` 是否存在于 `board` 中，存在返回 `True` 否则 `False`

**字符串的回溯算法**：字符串问题的特殊之处在于，字符串的拼接是产生新对象，而 `list` 对象是直接修改对象。

递归函数 `dfs(x, y, k)` 代表以网格的位置 `(x, y)` 出发，能否搜索到单词 `word[k:]`：

- 如果 `board[x][y] != s[k]` 返回 `False`；
- 如果 `k == len(s) - 1 and board[x][y] == s[k]` 返回 `True`；
- 否则继续递归所有相邻位置 `dfs(x + dx, y + dy, k + 1)`；

[17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)【中等】

【输入】一个仅包含数字 `2-9` 的字符串 `digits`

【输出】返回所有它能表示的字母组合

数字到字母的映射是电话九宫格的方式：`phoneMap = {'2': 'abc', '3': 'def', ..., '9': 'wxyz'}`

递归函数 `dfs(index, temp)` 表示当前决策第 `index` 个字母，`temp` 数组存放已经决定加入的字母：

- 如果 `index == len(digits)`，那么将当前的数组 `temp` 转化成字符串加入到答案数组中；
- 遍历当前位置对应的所有字母 `phoneMap[digits[index]]`，选择与不选择；

[784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)【中等】

【输入】一个字符串 `s`，将其中每个字母转变大小写，可以获得一个新的字符串

【输出】返回所有可能得到的字符串集合

递归函数 `dfs(index, temp)` 表示当前决策第 `index` 个字符，`temp` 数组存放已经在的字母：

- 如果 `index == len(s)`，将当前数组 `temp` 转化成字符串加入到答案数组中；

- 如果 `s[index].isalpha()` 是字母，那么分别添加大写和小写，再继续递归；
- 如果不是字母，那么直接添加，再递归；

[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)【中等】

【输入】整数 `n` 代表生成括号的对数

【输出】返回所有可能的 `n` 对括号组成的有效括号组合

递归函数 `dfs(temp, left, right)` 代表当前已决策的左右括号数量 `left`  和 `right`，以及数组 `temp`：

- 如果 `len(temp) == 2 * n` 将 `temp` 转化成字符串加入到答案数组；
- 如果左括号数量不大于 `n`，可以放一个左括号；如果右括号数量小于左括号数量，可以放一个右括号；

### 游戏问题

[51. N 皇后](https://leetcode.cn/problems/n-queens/)【困难】

[37. 解数独](https://leetcode.cn/problems/sudoku-solver/)【困难】

[488. 祖玛游戏](https://leetcode.cn/problems/zuma-game/)【困难】

[529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)【困难】

