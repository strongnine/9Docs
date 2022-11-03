## 二分问题

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/)【简单】

【输入】升序整型数组 `nums` 和目标值 `target`

【输出】返回 `target` 所在的下标，如果不存在返回 `-1`

二分，向下取整计算中点 `mid = (right - left) // 2 + left`：：

- 求中点 `mid = (right - left) // 2 + left`；
- 判断：循环条件 `left <= right`
  - 如果 `nums[mid] < target`，`left = mid + 1`；
  - 如果 `nums[mid] > target`，`right = mid - 1`；
  - 如果 `nums[mid] == target`，`return mid`；
- 如果找不到，就会因不满足循环条件跳出循环，按照题意 `return -1`。

[367. 有效的完全平方数](https://leetcode-cn.com/problems/valid-perfect-square/)【简单】

【输入】正整数 `num`

【输出】判断 `num` 是否为一个完全平方数

二分，以 `[1, num]` 作为区间，向下取整计算中点 `mid = (right - left) // 2 + left`：

- 判断：循环条件 `left <= right`，`square = mid * mid`
  - `square < num`，`left = mid + 1`；
  - `square > num`，`right = mid - 1`；
  - `square == num`，`return True`；

[374. 猜数字大小](https://leetcode-cn.com/problems/guess-number-higher-or-lower/)【简单】

【输入】数字 `n` 代表所选择的数字是在区间 `[1, n]` 中

【输出】猜出所选的数字 `pick` 是多少

有一个接口 `int guess(int num)`，向下取整计算中点 `mid = (right - left) // 2 + left`：：

- 如果 `pick < num`，返回 `-1`；
- 如果 `pick > num`，返回 `1`：
- 如果 `pick == num`，返回 `0`；

二分，以 `[1, n]` 作为区间：

- 判断：循环条件 `left <= right`：
  - `guess(mid) == 1`，`left = mid + 1`；
  - `guess(mid) == -1`，`right = mid - 1`；
  - `guess(mid) == 0`，`return mid`；

[441. 排列硬币](https://leetcode-cn.com/problems/arranging-coins/)【简单】

【输入】硬币数量 `n`

【输出】可以形成完整阶梯行的总行数 `row`。换句话说，即小于 `n` 的最大数列和 $1+2+3+\cdots+\text{row}$

二分，以 `[1, n]` 作为区间，向上取整计算中点 `mid = (right - left + 1) // 2 + left`

判断：循环条件 `left < right`：

- 如果 `mid * (mid + 1) <= 2 * n`，`left = mid`；
- 否则 `right = mid - 1`

[35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)【简单】

判断：循环条件 `left <= right`：

- 如果 `nums[mid] < target`，`left = mid + 1`；
- 否则有 `nums[mid] >= target`，`right = mid - 1`；

根据 if 的判断条件，`left` 左边的值一直保持小于 `target`，`right` 右边的值一直保持大于等于 `target`，而且 `left` 最终一定等于 `right + 1`。在循环结束之后，`left` 左边的部分全部小于 `target`，结尾的位置为 `right`；`right` 右边的部分全部大于等于 `target`，并且开始的位置是 `left`。因此最终的答案一定是 `left`. 

[278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)【简单】

判断：循环条件 `left < right`，下取整求 `mid`

- 如果 `isBadVersion(mid)`，`right = mid`；
- 否则，`left = mid + 1`；

[剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)【简单】

【输入】非递减数组 `nums` 以及目标数 `target`

【输出】返回 `target` 在 `nums` 中的出现次数

通过设定 `lower` 为 `True` 或者 `False`，以下二分查找代码可以查找在数组 `nums` 中第一个 `target` 的位置，以及第一个大于 `target` 的位置：

```python
class Solution:
    def binarySearch(self, nums, target, lower):
        n = len(nums)
        ans = n
        left, right = 0, n - 1
        while left <= right:
            mid = (right - left) // 2 + left
            if (nums[mid] > target) or (lower and nums[mid] >= target):
                right = mid - 1
                ans = mid
            else:
                left = mid + 1
        return ans

    
    def search(self, nums: List[int], target: int) -> int:
        leftIdx = self.binarySearch(nums, target, True)
        rightIdx = self.binarySearch(nums, target, False) - 1
        if (leftIdx <= rightIdx) and (rightIdx < len(nums)) and (nums[leftIdx] == target) and (nums[rightIdx] == target):
            return rightIdx - leftIdx + 1
        return 0
```

[1337. 矩阵中战斗力最弱的 K 行](https://leetcode-cn.com/problems/the-k-weakest-rows-in-a-matrix/)【简单】

【输入】大小为 `m * n` 的矩阵 `mat` 以及一个 `k`，`mat` 中的每一行的都包含一定数量的 1 和 0，并且 1 总是在 0 前

【输出】返回矩阵中包含 1 最少的 `k` 行的索引；如果两行包含 1 的数量一样那么行数小的排在前面

对于每一行，使用二分去查找最后一个 1 的位置 `pos`，那么该行包括的 1 的数量为 `pos + 1`。

然后把每一行包含 1 的数量以及该行的索引构成的元组建立小根堆，然后不断地弹出堆顶，将堆顶元素里的行索引放入答案数组。

- 对于所有的行 `i`：
  - 二分区间 `[0, n - 1]`，判断条件 `left <= right`，向下取整求 `mid`
  - 如果 `mat[i][mid] == 0`，`right = mid - 1`；
  - 否则 `mat[i][mid] == 1`，`left = mid + 1`， `pos = mid`；
- 将计算得到的 `[pos + 1, i]` 加入到一个列表 `power` 中；
- 小根堆：Python 中小根堆化为 `heapq.heapify(power)`，弹出堆顶为 `heapq.heappop(power)`；

[剑指 Offer II 069. 山峰数组的顶部](https://leetcode-cn.com/problems/B1IidL/)【简单】[852. 山脉数组的峰顶索引](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/)【中等】

【输入】山峰数组 `arr`，长度 `arr.length >= 3`

【输出】返回其中任何的一个山峰的位置

二分区间 `[1, n - 2]`，判断条件 `left <= right`，向下取整求 `mid`

- 如果 `arr[mid] > arr[mid + 1]`，`right = mid - 1`，`ans = mid`；
- 否则 `arr[mid] <= arr[mid + 1]`，`left = mid + 1`

[475. 供暖器](https://leetcode-cn.com/problems/heaters/)【中等】

【输入】房屋 `houses` 和供暖器 `heaters` 的位置

【输出】返回供暖器可以覆盖所有房屋的最小加热半径

- 将供暖器 `heaters` 排序；
- 对于每一个房屋 `house`：
  - 找到满足 `heaters[i] <= house` 的最大下标 `i`：
  - 假如 `heaters[0] > house` 时，`i = -1`，对应距离设为 `inf`；
  - 令 `j = i + 1`，则 `j` 是满足 `heaters[j] > house` 的最小下标；
  - 当 `heaters[n - 1] <= house` 时，`j = n`，对应距离设为 `inf`；
  - 离 `house` 最近的供暖器为 `heaters[i]` 或者 `heaters[j]`，这两个供暖器与 `house` 距离的最小值，即为考虑当前 `house` 时的最小加热半径；
  - 如果当前 `house` 的最小加热半径大于全局最小加热半径，更新全局最小加热半径；

[528. 按权重随机选择](https://leetcode-cn.com/problems/random-pick-with-weight/)【中等】

【输入】正整数数组 `w`

【输出】实现函数 `pickIndex`，随机从范围 `[0, w.length - 1]` 内返回一个下标。选取下标 `i` 的概率为 `w[i] / sum(w)`

- 构造前缀和数组 `sum`，长度为 `n + 1`，并且 `sum[i] = sum[i - 1] + w[i - 1]`；
- 选取阈值 `t` 为 `[1, sum[n - 1]]` 的整数
- 二分区间 `[1, n - 1]`，判断条件 `left < right`，向下取整求 `mid`；
  - 如果 `sum[mid] >= t`，`right = mid`；
  - 否则，`left = mid + 1`；
- 最终的答案为 `right - 1`；

[611. 有效三角形的个数](https://leetcode-cn.com/problems/valid-triangle-number/)【中等】

【输入】包含非负整数数组 `nums`

【输出】返回可以组成三角形的三元组的个数

- 将数组 `nums` 进行升序排序；
- `for i in range(n):`
  - `for j in range(i + 1, n):`
    - 二分区间 `[j + 1, n - 1]`，判断条件 `left <= right`，向下取整求 `mid`，`k = j`；
    - 如果 `nums[mid] < nums[i] + nums[j]`，`left = mid + 1`，`k = mid`；
    - 否则 `right = mid - 1`
  - 把答案累加 `ans += k - j`

[29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)【中等】

【输入】两个整数，被除数 `dividend` 和除数 `divisor`

【输出】在不使用乘法、除法、mod 运算下，返回 `dividend / divisor` 所得到的商的整数部分

> 注意：本题只能存储 32 位整数，如果结果溢出则返回 `2 ** 31 - 1`

根据题意有 `ans * divisor >= dividend > (ans + 1) * divisor`，因此可以使用二分法找到最大的满足 `ans * divisor >= dividend` 的 `ans`。

题目限制不能够使用乘法，可以使用「快速乘」来得到 `ans * divisor` 的值。

> 「快速乘」与「快速幂」类似，前者通过加法实现乘法，后者通过乘法实现幂运算。「快速幂」题目：[50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)，将里面的乘法运算改成加法运算就是「快速乘」

- 定义最小值和最大值：`INT_MIN, INT_MAX = -2 ** 31, 2 ** 31 - 1`；

- 判断特殊情况：

  - 如果被除数为最小值 `dividend = INT_MIN`：如果被除数为 1 返回 `INT_MIN`，如果被除数为 -1 返回 `INT_MAX`；
  - 如果被除数为 0，返回 0；

  - 如果除数为最小值 `divisor = INT_MIN`，

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)【中等】

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)【中等】

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)【中等】

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)【中等】

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)【中等】

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)【困难】

[162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)【中等】

[220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)【中等】

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)【中等】

[274. H 指数](https://leetcode-cn.com/problems/h-index/)【中等】

[275. H 指数 II](https://leetcode-cn.com/problems/h-index-ii/)【中等】

[1818. 绝对差值和](https://leetcode-cn.com/problems/minimum-absolute-sum-difference/)【中等】

[1838. 最高频元素的频数](https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/)【中等】

[1894. 找到需要补充粉笔的学生编号](https://leetcode-cn.com/problems/find-the-student-that-will-replace-the-chalk/)【中等】

[786. 第 K 个最小的素数分数](https://leetcode-cn.com/problems/k-th-smallest-prime-fraction/)【中等】

[911. 在线选举](https://leetcode-cn.com/problems/online-election/)【中等】

[981. 基于时间的键值存储](https://leetcode-cn.com/problems/time-based-key-value-store/)【中等】

[1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)【中等】

[1011. 在 D 天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/)【中等】

[1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)【中等】

[1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)【中等】

[1482. 制作 m 束花所需的最少天数](https://leetcode-cn.com/problems/minimum-number-of-days-to-make-m-bouquets/)【中等】

[352. 将数据流变为多个不相交区间](https://leetcode-cn.com/problems/data-stream-as-disjoint-intervals/)【困难】

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)【困难】

[354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)【困难】

[363. 矩形区域不超过 K 的最大数值和](https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/)【困难】

[778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/)【困难】

[1707. 与数组中元素的最大异或值](https://leetcode-cn.com/problems/maximum-xor-with-an-element-from-array/)【困难】

[1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)【困难】

[1751. 最多可以参加的会议数目 II](https://leetcode-cn.com/problems/maximum-number-of-events-that-can-be-attended-ii/)【困难】

## 双指针问题

## 背包问题

## 股票问题

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

## 岛屿问题

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



## 排列、组合、子集问题

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

[93. 复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)【中等】

【输入】只包含数字的字符串 `s`

【输出】输出所有可能的有效 IP 地址

> 有效 IP 地址：由四个 `[0, 255]` 之间的整数组成，例如 `0.1.2.201`, `192.168.1.1`, `255.255.255.255`

递归函数 `dfs(s, segId, segStart, temp)` 表示从 `s[segStart]` 开始，已经搜索完前 `segId` 个字段，其中 $\text{segId}\in\{0,1,2,3,4\}$，`temp` 数组保存全部 4 个字段：

- 如果当前已经到字符串的末尾 `segStart == len(s)`：
  - 如果已经找到了 4 个字段 `segId == 4`，那么将 `temp` 转化成符合格式的 IP 地址放到答案数组中；
  - 否则代表当前找的不是一个满足要求的，那么递归返回；

- 【剪枝 1】如果剩余的字符串长度 `restChar` 小于剩余要找的字段 `segId` 或者大于 `restId * 3` 那么进行剪枝（直接递归返回）；
- 遍历当前位置往后的 3 个长度的字符串 `segEnd in range(segStart, min(len(s), segStart + 3))`：
  - 由于不能含有前导零：如果字段长度为 1 并且当前字段为 0，那么只能单独成为一个字段；
  - 如果字段的范围在 `(0, 255]` 区间内，那么继续递归；
  - 【剪枝 2】否则进行剪枝；


具体的代码可以参考我的题解：[【回溯 + 剪枝】回溯实际上就是对一棵树的深度优先遍历](https://leetcode.cn/problems/restore-ip-addresses/solution/by-strongnine-9-yl4g/)

[733. 图像渲染](https://leetcode.cn/problems/flood-fill/)【简单】

【输入】大小为 `(m, n)` 的二维数组 `image`，着色位置 `sr, sc` 和颜色 `newColor`，`image[i][j]` 表示该点的像素值；`sr, sc` 表示从位置 `image[sr][sc]` 开始着色 `newColor`

【输出】返回经过上色渲染之后的图像

> 上色渲染：将上下左右所有颜色相同的位置都变成 `newColor`

递归函数 `dfs(image, x, y, preColor, newColor)`：

- 先记录 `image[sr][sc]` 的颜色 `preColor`，如果 `preColor` 和要修改的颜色 `newColor` 一样，那么不用递归；
- 先把当前颜色改成新颜色 `image[x][y] = newColor`，然后分别搜索上下左右 4 个方向，如果颜色是 `preColor` 则继续递归修改；

[130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)【中等】

【输入】大小为 `(m, n)` 的由若干 `X` 和 `O` 组成的二维数组 `board`

【输出】不输出任何内容，直接原地修改（将所有被 `X` 围绕的 `O` 都变成 `X`）

实际上，任何不与边界相连的 `O` 都是被 `X` 围绕的。

递归函数 `dfs(board, x, y)`，每寻找到一个边界的 `O` 就开始递归：

- 将与边界相连的 `O` 都标记为 `A`；
- 递归完成之后，如果 `board[x][y] == 'A'` 代表与边界相连，那么重新标记为 `O`；如果 `board[x][y] == 'O'` 代表没有与边界相连，那么重新标记为 `X`；

> **字符串的回溯算法**：字符串问题的特殊之处在于，字符串的拼接是产生新对象，而 `list` 对象是直接修改对象。

[79. 单词搜索](https://leetcode.cn/problems/word-search/)【中等】

【输入】大小为 `(m, n)`  的数组 `board`，一个字符串 `word`

【输出】返回 `word` 是否存在于 `board` 中，存在返回 `True` 否则 `False`

递归函数 `dfs(board, word, x, y, k, used)` 代表以网格的位置 `(x, y)` 出发，能否搜索到单词 `word[k:]`：

- 如果 `board[x][y] != word[k]` 返回 `False`；否则如果当前到了字符串末尾 `k == len(word) - 1` 则返回 `True`；
- 标记当前位置被使用 `used[x][y] = True`，遍历所有相邻位置，如果找到某一个位置有路径可以组成 `word`，则返回 `True`；
- 否则把当前位置重新标记为未使用 `used[x][y] = False `，返回 `False`；

具体的代码，可以参考题解：[回溯算法：就是深度优先搜索](https://leetcode.cn/problems/word-search/solution/by-strongnine-9-xl74/)

[17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)【中等】

【输入】一个仅包含数字 `2-9` 的字符串 `digits`

【输出】返回所有它能表示的字母组合

数字到字母的映射是电话九宫格的方式：`phoneMap = {'2': 'abc', '3': 'def', ..., '9': 'wxyz'}`

递归函数 `dfs(digits, idx, temp)` 表示当前决策第 `idx` 个字母，`temp` 数组存放已经决定加入的字母：

- 如果 `idx == len(digits)`，那么将当前的数组 `temp` 转化成字符串加入到答案数组中；
- 遍历当前位置对应的所有字母 `phoneMap[digits[index]]`，添加进 `temp` 里，继续递归；

具体的代码，可以参考题解：[回溯算法：深度优先搜索](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/solution/by-strongnine-9-ihfc/)

[784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/)【中等】

【输入】一个字符串 `s`，将其中每个字母转变大小写，可以获得一个新的字符串

【输出】返回所有可能得到的字符串集合

递归函数 `dfs(s, index, temp)` 表示当前决策第 `index` 个字符，`temp` 数组存放已经在的字母：

- 如果 `index == len(s)`，将当前数组 `temp` 转化成字符串加入到答案数组中；
- 如果 `s[index].isalpha()` 是字母，那么分别添加大写和小写，再继续递归；
- 如果不是字母，那么直接添加，再递归；

> 字母转换成大小写的方式：`char.upper()` 和 `char.lower()`. 

[22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)【中等】

【输入】整数 `n` 代表生成括号的对数

【输出】返回所有可能的 `n` 对括号组成的有效括号组合

递归函数 `dfs(temp, left, right)` 代表当前已决策的左右括号数量 `left`  和 `right`，以及数组 `temp`：

- 如果 `len(temp) == 2 * n` 将 `temp` 转化成字符串加入到答案数组；
- 如果左括号数量小于 `n`，可以放一个左括号；
- 如果右括号数量小于左括号数量，可以放一个右括号；

## 游戏问题

[51. N 皇后](https://leetcode.cn/problems/n-queens/)【困难】

[37. 解数独](https://leetcode.cn/problems/sudoku-solver/)【困难】

[488. 祖玛游戏](https://leetcode.cn/problems/zuma-game/)【困难】

[529. 扫雷游戏](https://leetcode.cn/problems/minesweeper/)【困难】

