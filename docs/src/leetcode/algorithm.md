# 算法总结

## 二分查找（Binary Search）

王争老师的二分查找循环写法：

```c++
int bsearch(vector<int> a, int n, int value) {
    int low = 0;
    int high = n - 1;
    
    while (low <= high) {
        int mid = low + (hight - low) / 2;
        if (a[mid] == value) {
            return mid;
        } else if (a[mid] < value) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    return -1;
}
```

递归写法：

```c++
int bsearch(vector<int> a, int n, int val) {
    return bsearchInternally(a, 0, n - 1, val);
}

int bsearchInternally(vector<int>, int low, int high, int value) {
    if (low > high) return -1;
    
    int mid = low + ((high - low) >> 1);
    if (a[mid] == value) {
        return mid;
    } else if (a[mid] < value) {
        return bsearchInternally(a, mid+1, high, value);
    } else {
        return bsearchInternally(a, low, mid-1, value);
    }
}
```

>  liweiwei 的版本下，循环条件是 `left < right`，他的写法中，退出循环时一定有 `left == right` 成立。

二分查找的时间复杂度是 $\mathcal{O}(\log n)$，查找数据的效率非常高。

二分查找的局限性：

- 依赖顺序表结构，也就是数组；
- 只针对有序数据。如果是对于静态数据（没有频繁插入和删除），可以用一次排序，多次二分查找，来均摊排序的成本；
- 数据量太小不适合二分查找；
- 数据量太大也不适合用二分。主要原因在于二分查找依赖于数组，而数组需要的是连续的内存空间，如果数组太大，难以找到连续的内存空间。；

二分查找的原理及其简单，但是想要写出没有 Bug 的二分查找并不容易。

> 尽管第一个二分查找算法在 1946 年出现，然而第一个完全正确的二分查找算法实现直到 1962 年才出现。
>
> —— 《计算机程序设计艺术》唐纳德 • 克努特（Donald E.Knuth）

4 种常见二分查找变形问题：

- 查找第一个值等于给定值的元素；
- 查找最后一个值等于给定值的元素；
- 查找第一个大于等于给定值的元素；
- 查找最后一个小于等于给定值的元素；

**变体一：查找第一个值等于给定值的元素**

```c++
int bsearch(vector<int> a, int n, int value) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        if (a[mid] > value) {
            high = mid - 1;
        } else if (a[mid] < value) {
            low = mid + 1;
        } else {
            if ((mid == 0) || (a[mid - 1] != value)) return mid;
            else high = mid - 1;
        }
    }
    return -1;
}
```

对于第 11 行代码：

- 如果 `mid == 0`，那这个元素是第一个元素，肯定是我们要找的；
- 如果 `mid != 0` 并且前面一个元素 `a[mid-1]` 不等于要找的值 `value`，那这个元素就是我们要找的；
- 如果 `a[mid]` 前面的元素也是 value，那么此时的 `a[mid]` 肯定不是第一个等于给定值 `value` 的元素，让 `high = mid - 1`；

## 最高有效位

如果正整数 $y$ 是 2 的整数次幂，则 $y$ 的二进制表示中只有最高位为 1，其余都是 0，因此 $y\& (y-1)=0$. 如果 $y\le x$，则称 $y$ 为 $x$ 的「最高有效位」。

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用这样的一个方式，动态地维护最高有效位 `highBit`，然后算出所有小于给定整数 $n$ 的整数二进制表示包含 1 的数量。

## 最低有效位

对于正整数 $x$，将其二进制表示右移一位，等价于将其二进制表示的最低位去掉，得到的数是 $\lfloor\frac{x}{2}\rfloor$，如果 $bits[\lfloor\frac{x}{2}\rfloor]$ 的值已知，则可以得到 $bits[x]$ 的值：

- 如果 $x$ 是偶数，则 $bits[x]=bits[\lfloor\frac{x}{2}\rfloor]$；
- 如果 $x$ 是奇数，则 $bits[x] = bits[\lfloor\frac{x}{2}\rfloor]+1$.

上述两种情况可以合并成：$bits[x] $ 的值等于 $bits[\lfloor\frac{x}{2}\rfloor]$ 的值加上 $x$ 除以 2 的余数。

由于 $\lfloor\frac{x}{2}\rfloor$ 可以通过 $x\gg1$ 得到，$x$ 除以 2 的余数可以通过 $x \& 1$ 得到，因此有：$bits[x]=bits[x\gg1]+(x\&1)$.

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用最低有效位，遍历从 1 到 $n$ 的每个正整数 $i$，计算 $bits$ 的值，最终得到的数组 $bits$ 即为答案。

## Brian Kernighan 算法

记 $f(x)$ 表示 $x$ 和 $x-1$ 进行「与」运算所得的结果（即 $f(x)=x\&(x−1)$），那么 $f(x)$ 恰为 $x$ 删去其二进制表示中最右侧的 1 的结果。参考 LeetCode 题目 [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)。

例如 $x=10001000$，$x-1=10000111$，那么 $x\&(x-1)=10000000$. 

利用 Brian kernighan 算法计算出一个数的二进制表示有多少个 1 的方法如下。不断让 $s=f(s)$，直到 $s=0$。每循环一次 $s$ 都会删除二进制表示中最右侧的 1，最终的循环次数即为 $s$ 二进制表示中的 1 的数量。

## 广度优先搜索（Breadth-First Search）

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

## 深度优先搜索（Depth-First Search）

C++ 深度优先搜索的框架

```c++
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

## 回溯算法（Backtrack）

适用问题：

- 解决一个问题有多个步骤
- 每个步骤有多种方法
- 需要找出所有的方法

原理：在一棵树上的**深度优先遍历**
