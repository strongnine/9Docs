LeetCode 刷题会用到的知识点，包括语法、函数等等；

各种数据类型的范围：

- `int`: $[-2147483648, 2147483647]$，大约等于 $-2\times 10^{9}\sim 2\times 10^{9}$.

位运算符：

| 运算符 |描述 | 实例 |
| ------ | ---- | ---- |
| &      | 按位「与」运算 |      |
| \|     | 按位「或」运算 |      |
| ^      | 按位「异或」运算 |      |
| ~      | 按位「取反」运算 |      |
| <<     | 二进制左移运算 |      |
| >>    | 二进制右移运算 |      |

## 数据结构

### 向量（vector）

```cpp
// 给向量 x 分配 size 个 value 值
x.assign(int size, int value);	
// 给向量 x 分配从迭代器初始位置到最终位置的值
x.assign(InputIterator first, InputIterator last);	

// 后面添加元素
x.push_back();
// 将最后元素弹出
x.pop_back();
```

### 栈（stack）





### 队列（queue）

```c++
// 生成方式
std::queue<std::string> q;
std::queue<std::string> q {arr};
// 操作
// 返回第一个元素的引用，如果 q 为空，返回值是未定义的
q.front();
// 返回最后一个元素的引用
q.back();
// 在尾部添加一个元素的副本
q.push(const T& obj);
// 在尾部生成对象
q.emplace();
// 删除第一个元素
q.pop();
// 返回元素个数
q.size();
// 判断是否为空
q.empty();
```





### 链表（linked-list）

在 LeetCode 中链表节点的定义为：

```c++
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
```

**哑节点（Dummy node）：**在操作链表时常用的技巧，它的 next 指针指向链表头节点。好处是不用对头节点进行特殊判断。

### 常用函数

`__builtin_popcount()`：返回输入数据，二进制中「1」的个数，只对 `int` 类型；

如果想要对 `long int` 和 `long long` 类型使用，可以分别用 `__builtin_popcountl()` 和 `__builtin_popcountll`

`lower_bound(ForwardIterator first, ForwardIterator last, const T& val)`：返回一个迭代指针，该指针指向在 `[first, last)` 中不小于 `val` 的第一个元素。

`isalpha(c)`：判断 `c` 是否为一个字母。

`tolower(c)`：将字母转为小写字母；`toupper(c)`：将字母转为大写字母；

`to_string(c)`：将 `c` 转化为字符；

## 算法

### 最高有效位

如果正整数 $y$ 是 2 的整数次幂，则 $y$ 的二进制表示中只有最高位为 1，其余都是 0，因此 $y\& (y-1)=0$. 如果 $y\le x$，则称 $y$ 为 $x$ 的「最高有效位」。

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用这样的一个方式，动态地维护最高有效位 `highBit`，然后算出所有小于给定整数 $n$ 的整数二进制表示包含 1 的数量。

### 最低有效位

对于正整数 $x$，将其二进制表示右移一位，得到 $$

### Brian Kernighan 算法

记 $f(x)$ 表示 $x$ 和 $x-1$ 进行「与」运算所得的结果（即 $f(x)=x\&(x−1)$），那么 $f(x)$ 恰为 $x$ 删去其二进制表示中最右侧的 1 的结果。参考 LeetCode 题目 [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)。

例如 $x=10001000$，$x-1=10000111$，那么 $x\&(x-1)=10000000$. 

利用 Brian kernighan 算法计算出一个数的二进制表示有多少个 1 的方法如下。不断让 $s=f(s)$，直到 $s=0$。每循环一次 $s$ 都会删除二进制表示中最右侧的 1，最终的循环次数即为 $s$ 二进制表示中的 1 的数量。

### 广度优先搜索（Breadth-First Search）

广度优先搜索 C++ 模板：

```c++
private:
	static constexpr int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

public:
	vector<vector<int>> BFS(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size(0);
        vector<vector<int>> seen(m, vector<int>(n));
        queue<pair<int, int>> q;

        // 广度优先搜索
        while (!q.empty()) {
            auto [i, j] = q.front();
            q.pop();
			for (int d = 0; d < 4; ++d) {
                int ni = i + dirs[d][0];
                int nj = j + dirs[d][1];
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && !seen[ni][nj]) {
                    // ...;	// 写某些具体的操作
                    q.emplace(ni, nj);
                    seen[ni][nj] = 1;
                }
            }
        }
        return ...;			// 返回结果
    }

```

Python3 模板

```python
def BFS(self, matrix: List[List[int]]) -> ...:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    m, n = len(matrix), len(matrix[0])
    q = collections.deque(...);		# 队列
    seen = set(...)					# 将见过的位置坐标放进集合里面
    
    while q:
        i, j = q.popleft()
        for d in range(4):
            ni = i + dirs[d][0]
            nj = j + dirs[d][1]
            if (0 <= ni < m) and (0 <= nj < n) and ((ni, nj) not in seen):
                # ...				# 写某些具体操作
                q.append((ni, nj))
                seen.add((ni, nj))
    return ...						# 返回结果 
```

### 深度优先搜索（Depth-First Search）

```c++
// C++ 深度优先搜索 的框架
vector<int> temp;
void dfs(int cur, int n) {
    if (cur == n + 1) {
        // 记录答案
        // ...
        return;
    }
    // 考虑选择当前位置
    temp.push_back(cur);
    dfs(cur + 1, n, k);
    temp.pop_back();
    // 考虑不选择当前位置
    dfs(cur + 1, n, k);
}
```

### 回溯算法（Backtrack）

适用问题：

- 解决一个问题有多个步骤
- 每个步骤有多种方法
- 需要找出所有的方法

原理：在一棵树上的**深度优先遍历**





## 做题记录

时间复杂度：$O(n)$，空间复杂度：$O(n)$.

### 二分搜索（Binary Search）

二分查找的题目，就看 liweiwei 的题解就行了：[写对二分查找不能靠模板，需要理解加练习 （附练习题）](https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/)

一般情况下，边界设置为 `left = mid + 1` 与 `right = mid`，这个时候中点是下取整，即偏向于左边取：`mid = (right - left) / 2 + left`。

> 当看到边界设置行为是 `left = mid` 与 `right = mid - 1` 的时候，需要将 `mid = (right - left + 1) / 2 + left`，即调整为上取整，即偏向于右边取。

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/liang-ci-er-fen-cha-zhao-by-strongnine-9-04l4/)；

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-rmzn/)；

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-toku/)；

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-k84i/)；

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-mszd/)；

### 栈

[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)，我的[题解](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/solution/yong-liang-ge-zhan-shi-xian-dui-lie-by-s-0dtx/)；

[剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)，我的[题解](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/solution/wei-hu-liang-ge-zhan-lai-shi-xian-by-str-gyca/)；

[剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)，我的[题解](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/san-chong-fang-fa-jie-jue-fan-xiang-da-y-irt5/)；

### 树

前序遍历、中序遍历、后序遍历基本写法：

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)，

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)，

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)，

关于前中后序三种遍历，极客时间王争的课程给了三个对应的递推代码

```c++
// 前序遍历
void preOrder(Node* root) {
  if (root == null) return;
  print root // 此处为伪代码，表示打印root节点
  preOrder(root->left);
  preOrder(root->right);
}

// 中序遍历
void inOrder(Node* root) {
  if (root == null) return;
  inOrder(root->left);
  print root // 此处为伪代码，表示打印root节点
  inOrder(root->right);
}

// 后序遍历
void postOrder(Node* root) {
  if (root == null) return;
  postOrder(root->left);
  postOrder(root->right);
  print root // 此处为伪代码，表示打印root节点
}
```



### 双指针

**简单题：**

[350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)，我的[题解](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/solution/350-liang-ge-shu-zu-de-jiao-ji-shi-yong-nyhsl/)；

**中等题：**

[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)；



### 链表

[剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)，我的[题解](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/die-dai-he-di-gui-liang-chong-fang-fa-by-s3su/)；



### 动态规划（Dynamic Programming）

动态规划的 liweiwei 有一个关于买卖股票问题的题解：[暴力解法、动态规划（Java）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/bao-li-mei-ju-dong-tai-gui-hua-chai-fen-si-xiang-b/)，还有[股票问题系列通解（转载翻译）](https://leetcode-cn.com/circle/article/qiAgHn/)。

**简单题：**

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)，我的[题解](https://leetcode-cn.com/problems/maximum-subarray/solution/53-zui-da-zi-xu-he-dong-tai-gui-hua-by-s-csae/)；

**经典股票系列问题**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)，简单题，我的[题解](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/121-mai-mai-gu-piao-de-zui-jia-shi-ji-ji-54ir/)。

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)，中等题，

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)，困难题，

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)，困难题，

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)，中等题，

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)，中等题，

### 哈希

[350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)，我的[题解](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/solution/350-liang-ge-shu-zu-de-jiao-ji-shi-yong-nyhsl/)；



## 0/ 题解草稿
