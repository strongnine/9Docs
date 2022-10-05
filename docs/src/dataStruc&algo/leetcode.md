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

排列、组合、子集问题是学习回溯算法的一系列很好的问题。回溯算法就是在一棵树上的深度优先遍历（DFS）

[46. 全排列](https://leetcode.cn/problems/permutations/)【中等】

【输入】一个不包含重复数字的序列 `nums`

【输出】返回所有不重复的全排列（不考虑顺序）

递归函数 `backtrack(first, output)` 表示从左往右填到第 `first` 个位置，当前排列为 `output`：

- `first == n` 时，将 `output` 放入答案数组中，递归结束；
- 定义一个标记数组 `vis`，递归时遍历 `nums` 找到没被选择的数字填入 `output`，继续调用 `backtrack(...)` 函数；
- 回溯的时候要撤销对应位置的数以及标志，继续尝试其他没被标记过的数；

[47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)【中等】

【输入】一个包含重复数字的序列 `nums`

【输出】返回所有不重复的全排列（不考虑顺序）

递归函数 `backtrack(idx, perm)` 表示当前排列为 `perm`，下一个待填入的位置是 `idx`；

- 如果 `idx == n` 将 `perm` 放入答案数组中；
- 保证填第 `idx` 个数的时候重复数字只会被填入一次：对原数组排序，保证相同的数字都相邻，每次填入的数字一定是这个数所在重复数集合中「从左往右第一个未被填过的数字」；

> 注意：组合与排列的不同，排列问题是关注顺序的，而组合问题不关注。具体而言，对于数组 `[1, 2, 3]` 和 `[1, 3, 2]` 在组合问题中是一样的，而在排列问题中则不一样。

[39. 组合总和](https://leetcode.cn/problems/combination-sum/)【中等】

【输入】一个无重复元素的整数数组 `candidates` 和目标整数 `target`

【输出】返回 `candidates` 中可以使得数字和为 `target` 的所有不同组合所组成的列表（不考虑顺序）

递归函数 `dfs(target, combine, idx)` 表示当前在 `candidates` 数组的第 `idx` 位，还剩 `target` 要组合，已经组合的列表为 `combine`：

- 当 `target == 0` 时将当前组合 `combine` 加入到答案列表中；

- 当 `target < 0` 或者 `candidates` 全部遍历完，递归终止；
- 选择使用第 `idx` 个数 `dfs(target - candidates[idx], combine, idx)`；选择跳过不使用 `idx` 个数 `dfs(target, combine, idx + 1)`；

[40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)【中等】

【输入】一个含重复元素的整数数组 `candidates` 和目标整数 `target`

【输出】返回 `candidates` 中使得数字和为 `target` 的所有不同组合所组成的列表（不考虑顺序）

递归函数 `dfs(pos, rest)`，`pos` 表示我们当前递归到了数组 `candidates` 中的第 `pos` 个数，`rest` 表示我们还需要选择和为 `rest` 的数放入列表作为一个组合：

- 当 `rest == 0` 时将当前组合 `combine` 加入到答案列表中；
- 当 `target < 0` 或者 `candidates` 全部遍历完，递归终止；
- 选择使用第 `idx` 个数 `dfs(target - candidates[idx], combine, idx)`；选择跳过不使用 `idx` 个数 `dfs(target, combine, idx + 1)`；

为了避免出现重复的组合：

- 使用一个 `HashMap` 统计数组 `candidates` 中每个数出现的次数，将结果放入一个列表 `freq` 中，方便后续的递归使用。列表 `freq` 的长度即为数组 `candidates` 中不同数的个数，其中每一项对应着哈希映射中的一个键值对，即某个数以及它出现的次数；
- 递归时，对于当前的第 `pos` 个数，它的值为 `freq[pos][0]`，出现的次数为 `freq[pos][1]` 那么我们可以调用 `dfs(pos + 1, rest - i * freq[pos][0])`。即我们选择了这个数 `i` 次，并且 `i` 不能大于这个数出现的次数，`i * freq[pos][0]` 也不能大于 `rest`，同时需要将 `i` 个 `freq[pos][0]` 放入列表中；

[77. 组合](https://leetcode.cn/problems/combinations/)【中等】

【输入】两个整数 `n` 和 `k`

【输出】返回范围 `[1, n]` 中所有的可能的 `k` 个数的组合

递归函数 `dfs(cur, n, k)` 表示当前在位置 `cur`，数组范围为 `n` 要选择 `k` 个数字进行组合，`temp` 数组用于记录当前已经选择的数字：

- 剪枝：如果 `temp` 的长度加上区间 `[cur, n]` 的长度小于 `k`，那么不可能构造出长度为 `k` 的 `temp`；
- 如果 `temp`  的长度已经等于 `k`，那么记录当前的组合；
- 考虑当前位置，则 `temp.append(cur)`；不考虑当前位置，则直接对下一个位置进行决策 `dfs(cur + 1, n, k)`；

[78. 子集](https://leetcode.cn/problems/subsets/)【中等】

【输入】一个不含重复元素的整数数组 `nums`

【输出】返回该数组所有可能的子集（不考虑顺序）

递归函数 `dfs(cur, nums, temp)` 代表当前决策数组 `nums` 中的第 `cur` 个位置，临时数组 `temp` 中为决定放进子集的数字：

- 如果当前已经决策完所有的数字 `cur == len(nums)`，则把 `temp` 作为答案放进答案数组 `self.ans.append(temp.copy())`；
- 将当前位置加入临时数组 `temp`，则 `temp.append(nums[cur])`，再对下一个位置进行决策；否则不加入，直接对下一个位置进行决策 `dfs(cur + 1, nums, temp)`；

[90. 子集 II](https://leetcode.cn/problems/subsets-ii/)【中等】

【输入】包含重复元素的整数数组 `nums`

【输出】返回该数组所有不重复的子集（不考虑顺序）

避免重复的方法：先对数组 `nums` 进行排序，保证相同的数字都在一起。如果没有选择上一个数，并且当前数字与上一个数相同，则跳过当前生成的子集。

递归函数 `dfs(choosePre, cur, nums, temp)` 代表当前决策数组 `nums` 中的第 `cur` 个位置，临时数组 `temp` 中为决定放进子集的数字，`choosePre` 表示前一个数字是否被选择：

- 如果当前已经决策完所有的数字 `cur == len(nums)`，则把 `temp` 作为答案放进答案数组 `self.ans.append(temp.copy())`；
- 对于当前的位置 `cur`：
  - 如果 `!choosePre and cur > 0 and nums[cur - 1] == nums[cur]`，则直接 `return`
  - 否则，将当前位置加入临时数组 `temp`，则 `temp.append(nums[cur])`，再对下一个位置进行决策；否则不加入，直接对下一个位置进行决策 `dfs(False, cur + 1, nums, temp)`；

[60. 排列序列](https://leetcode.cn/problems/permutation-sequence/)【困难】

【输入】给定 `n` 和 `k`

【输出】返回集合 `[1, 2, 3,..., n]` 所有排列的第 `k` 个

这个题目可以很好地学习剪枝的思想，基本的思路是 [46. 全排列](https://leetcode.cn/problems/permutations/) 再加上剪枝。

剪枝：在每个分支里，根据还未选定的个数计算阶乘 $(n - 1 - index)!$ 来得到当前分支有多少个叶子结点：

- 如果 `k` 大于这个分支的叶子结点数，代表答案肯定不在这个分支里，直接跳过当前分支；
- 如果 `k` 小于这个分支的叶子结点数，代表答案肯定在这个分支里，继续递归进行剪枝；

递归函数 `dfs(index, path, used)` 表示当前决策第 `index` 个数，`path` 保存已经决策完的 `index - 1` 个数，`used` 用于标记已经使用过的数字。

- 如果 `index == self.n` 则代表 `path` 为要返回的第 `k` 个序列；
- 每一次 DFS 都要判断 `i = [1, 2,..., n]` 哪个数可以用：
  - 如果 `used[i] == False`
  - 计算 `cnt = factorial[n - 1 - index]`，多减一个 1 是因为当前已经在决策其中一个数；
    - 如果 `cnt < k` 则要对这个分支进行剪枝，令 `k -= cnt` 然后寻找下一个未被使用的数字，即 `used[i] == False` 的；
  - 否则 `cnt >= k` 代表答案在当前分支里，选择当前数字 `path.append(i)`，标记为已使用 `used[i] = Ture`，继续递归决策下一个数 `dfs(index + 1, path, used)`；

[93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)【中等】



[733. 图像渲染](https://leetcode.cn/problems/flood-fill/)【简单】



[130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)【中等】



[79. 单词搜索](https://leetcode.cn/problems/word-search/)【中等】

**字符串的回溯算法**：字符串问题的特殊之处在于，字符串的拼接是产生新对象，而 `list` 对象是直接修改对象。

[17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)【中等】



[784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)【中等】



[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)【中等】

### 游戏问题

[51. N 皇后](https://leetcode.cn/problems/n-queens/)【困难】

[37. 解数独](https://leetcode.cn/problems/sudoku-solver/)【困难】

[488. 祖玛游戏](https://leetcode.cn/problems/zuma-game/)【困难】

[529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)【困难】

