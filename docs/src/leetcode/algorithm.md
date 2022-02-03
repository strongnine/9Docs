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
- 数据量太大也不适合用二分。主要原因在于二分查找依赖于数组，而数组需要的是连续的内存空间，如果数组太大，难以找到连续的内存空间；

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
            if ((mid == 0) || (a[mid - 1] != value)) return mid;    // 第 11 行
            else high = mid - 1;
        }
    }
    return -1;
}
```

关键是第 11 行代码：

- 如果 `mid == 0`，那这个元素是第一个元素，肯定是我们要找的；
- 如果 `mid != 0` 并且前面一个元素 `a[mid-1]` 不等于要找的值 `value`，那这个元素就是我们要找的；
- 如果 `a[mid]` 前面的元素也是 value，那么此时的 `a[mid]` 肯定不是第一个等于给定值 `value` 的元素，让 `high = mid - 1`；

**变体二：查找最后一个值等于给定值的元素**

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
            if ((mid == n - 1) || (a[mid + 1] != value)) return mid;  // 第 11 行
            else low = mid + 1;
        }
    }
    return -1;
}
```

关键是第 11 行代码：

- 如果 `a[mid]` 是数组中的最后一个元素，那肯定是要找的值；
- 如果 `a[mid]` 的最后一个元素 `a[mid+1]` 不等于 `value`，那也说明是要找的最后一个值等于给定值的元素；
- 如果发现 `a[mid]` 的后面一个元素 `a[mid+1]` 也等于 `value`，说明当前的 `a[mid]` 并不是最后一个值等于给定值的元素，更新 `low = mid + 1`；

**变体三：查找第一个大于等于给定值的元素**

```c++
int bsearch(vector<int> a, int n, int value) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        if (a[mid] >= value) {
            if ((mid == 0) || (a[mid - 1] < value)) return mid;  // 第 7 行
            else high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return -1;
}
```

如果 `a[mid]` 小于给定的值 `value`，那要查找的值肯定在 `[mid+1, high]` 之间，所以更新 `low = mid + 1`；

对于第 7 行代码，如果 `a[mid]` 大于等于给定值 `value`：

- 如果 `a[mid]` 为第一个元素，或者前面一个元素小于要查找的值 `value`，那 `a[mid]` 就是我们要找的元素；
- 如果 `a[mid-1]` 也大于等于要查找的值 `value`，那么说明要查找的元素在 `[low, mid-1]` 之间，更新 `high = mid - 1`；

**变体四：查找最后一个小于等于给定值的元素**

```c++
int bsearch(vector<int> a, int n, int value) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        if (a[mid] > value) {
            high = mid - 1;
        } else {
            if ((mid == n - 1) || (a[mid + 1] > value)) return mid;  // 第 9 行
            else low = mid + 1;
        }
    }
    return -1;
}
```

如果 `a[mid]` 大于给定的值 `value`，那么要查找的值肯定在区间 `[low, mid-1]` 之间，所以更新 `high = mid - 1`；

对于第 9 行代码，如果 `a[mid]` 小于等于给定的值：

- 如果 `a[mid]` 为最后一个元素，或者其后面一个元素大于要查找的值 `value`，那么其肯定是要找的元素；
- 如果 `a[mid+1` 也小于等于要查找的值，那么说明要查找的元素在 `[low, mid+1]` 之间，更新 `low = mid + 1`；

## 二叉树的遍历

实际上，二叉树的前、中、后序遍历就是一个递归过程。写递归代码的关键，就是写出递推公式。写递归公式的关键，就是如果要解决问题 A，就假设问题 B、C 已经解决，然后再来看如何利用 B、C 来解决 A。

```c++
// 前序遍历的递推公式：
preOrder(r) = print r => preOrder(r->left) => preOrder(r->right);

// 中序遍历的递推公式：
inOrder(r) = inOrder(r->left) => print r => inOrder(r->right);

// 后序遍历的递推公式：
postOrder(r) = postOrder(r->left) => postOrder(r->right) => print r;
```

根据递推公式，可以写出三种遍历方式的伪代码：

```c++
// 前序遍历
void preOrder(TreeNode* root) {
  if (root == nullptr) return;
  print root // 此处为伪代码，表示打印 root 节点
  preOrder(root->left);
  preOrder(root->right);
}

// 中序遍历
void inOrder(TreeNode* root) {
  if (root == nullptr) return;
  inOrder(root->left);
  print root // 此处为伪代码，表示打印 root 节点
  inOrder(root->right);
}

// 后序遍历
void postOrder(TreeNode* root) {
  if (root == nullptr) return;
  postOrder(root->left);
  postOrder(root->right);
  print root // 此处为伪代码，表示打印 root 节点
}
```

由于每个节点最多会被访问两次，所以前中后序遍历的时间复杂度为 $\mathcal{O}(n)$。

## 广度优先搜索（Breadth-First Search）

广度优先搜索有三个重要的辅助变量：

- `visited` 用来记录顶点是否被访问了，避免顶点被重复访问；
- `queue` 队列，用来存储已经访问，但相连的顶点还没有被访问的顶点；
- `prev` 用来记录搜索路径。例如 `prev[w]` 存储的是，顶点 `w` 是从哪个前驱顶点遍历过来的。所以为了正向打印出路径，需要递归地进行。

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

时间复杂度：$\mathcal{O}(V+E)$，空间复杂度：$\mathcal{O}(V)$；$V, E$ 分别为有向图顶点和边的个数。

## 深度优先搜索（Depth-First Search）

深度优先搜索（DFS），最直观的例子就是「走迷宫」。

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

时间复杂度：$\mathcal{O}(E)$，空间复杂度：$\mathcal{O}(V)$；$V, E$ 分别为有向图顶点和边的个数。

## 动态规划（Dynamic Programming）

把问题分解为多个阶段，每个阶段对应一个决策。记录每一个阶段可达的状态合集（去掉重复的），然后通过当前阶段的状态集合，来推导下一个阶段的状态集合，动态地往前推进。

**0 - 1 背包问题**：对于一组不同重量、不可分割的物品，我们需要选择一些装入背包，在满足背包最大重量限制的前提下，背包中物品总重量的最大值是多少？

用一个二维数组 `state[n][w+1]` 来记录每个物品决策完可以到达的状态。

```c++
// weight: 物品重量，n: 物品个数，w: 背包可承载重量
int knapsack(vector<int> weight, int n, int w) {
    vector<vector<int>> states(n, vector<int>(w + 1)); // 默认值 0
    states[0][0] = 1;  // 第一行的数据要特殊处理，可以利用哨兵优化
    if (weight[0] <= w) {
        states[0][weight[0]] = 1;
    }
    for (int i = 1; i < n; ++i) {  // 动态规划状态转移
        for (int j = 0; j <= w; ++j) {  // 不把第 i 个物品放入背包
            if (states[i - 1][j] == 1) {states[i][j] = states[i - 1][j]};
        }
        for (int j = 0; j <= w - weight[i]; ++j) {  // 把第 i 个物品放入背包
          if (states[i - 1][j] == 1) {states[i][j + weight[i]] = 1;}  
        }
    }
    for (int i = w; i >= 0; --i) {  // 输出结果
        if (states[n - 1][i] == 1) {return i;}
    }
    return 0;
}
```

一般来说动态规划是一种空间换时间的方法。

## 回溯算法（Backtrack）

**0 - 1 背包**：

```c++
// 0 - 1 背包的回溯算法实现
vector<int> weight = {2, 2, 4, 6, 3};   //	物品重量
int w = 9;      // 	背包承受的最大重量
int maxW = INT_MIN;
void f(int i, int cw, int w) {  // 调用 f(0, 0)
    int n = weight.size()
    if (cw == w || i == n) {  // cw == c 表示装满了，i == n 表示物品都考察完了
        if (cw > maxW) {
            maxW = cw;
            return maxW;
        }
    }
    f(i + 1, cw, w);  // 选择不装第 i 个物品
    if (cw + weight[i] <= w) {
        f(i + 1, cw + weight[i], w);  // 选择装第 i 个物品
    }
}
```

这种写法的时间复杂度很高，可以使用「备忘录」来记录所有已经计算过的值，再次要用到的时候，就直接拿出来。

```c++
vector<int> weight = {2, 2, 4, 6, 3};     // 物品重量
int w = 9;
int maxW = INT_MIN;
vector<vector<int>> mem(n, vector<int>(w + 1));
void f(int i, int cw, int w) {
    int n = weight.size()
    if (cw == w || i == n) {
        if (cw > maxW) {
            maxW = cw;
            return;
        }
    }
    if (mem[i][cw]) {return;}  //  重复状态
    mem[i][cw] = true;   // 记录 (i, cw) 这个状态
    f(i + 1, cw);        // 选择不装第 i 个物品
    if (cw + weight[i] <= w) {
        f(i + 1, cw + weight[i]);  // 选择装第 i 个物品
    }
}
```

加了「备忘录」的方法跟动态规划的执行效率已经基本没有差别。

适用问题：

- 解决一个问题有多个步骤
- 每个步骤有多种方法
- 需要找出所有的方法

原理：在一棵树上的**深度优先遍历**



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

## 工程问题与算法题

工程上的问题比较开放，需要综合各种因素去选择数据结构和算法。我们要考虑到的可能有编码难度、维护成本、数据特征、数据规模等，只有最合适的方法，没有最好的方法。而算法题的背景、条件、限制都十分明确，只需要在规定的输入、输出下找到最优解就好了。

关于合理地选择使用哪种数据结构和算法，王争老师在极客时间上的课程《数据结构与算法之美》$^{[1]}$中总结了六条经验：

1. **时间、空间复杂度不能跟性能划等号**；
   - 复杂度不是执行时间和内存消耗的精确值；
   - 代码的执行时间有时不跟时间复杂度成正比【大 $\mathcal{O}$ 表示法有效的前提是在处理大规模数据的情况下才成立】；
   - 对于处理不同问题的不同算法，其复杂度大小没有可比性；
2. **抛开数据规模谈数据结构和算法都是「耍流氓」**；
3. **结合数据特征和访问方式来选择数据结构。**如何将一个背景复杂、开放的问题，通过细致的观察、调研、假设，理清楚要处理数据的特征与访问方式，这才是解决问题的重点；
4. **区别对待 IO 密集、内存密集和计算密集**；
5. **善用语言提供的类，避免重复造轮子**；
6. **千万不要漫无目的地过度优化。**要优化代码的时候，要先做 Benchmark 基准测试，避免想当然地换了更高效的算法，但是实际上性能是下降了的；

## 参考

[1] 《数据结构与算法之美》王争｜极客时间

[2] LeetCode 题解

