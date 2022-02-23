# 算法总结

## 排序算法

用有序度来表示一组数据的有序程度，用逆序度表示一组数据的无序程度。假设有 $n$ 个数据，要从小到大排列，完全有序的数据的有序度就是 $n(n-1)/2$，逆序度就是 0. 

### 归并排序（Merge Sort）

核心思想：先把数值从中间分成前后两个部分，然后对前后两个部分分别排序，再将排好的两个部分合并在一起，这样整个数组就都有序了。归并排序可以用递归的方式来实现，写递归代码的技巧：

- 分析得出递推公式；
- 找到终止条件；
- 将递推公式翻译成递归代码；

我根据王争老师的的课程[《数据结构与算法之美》第 12 讲](http://gk.link/a/11bHN)中给的伪代码，翻译出了一个 Python 版本的：

```python
# 递推公式：
# merge_sort(p...r) = merge(merge_sort(p...q), merge_sort(q+1...r))
# 
# 终止条件：
# p >= r 不用再继续分解

# 归并排序算法，A 是数组，n 表示数组大小
def merge_sort(A, n):
    return merge_sort_c(A, 0, n-1)

# 递归调用函数
def merge_sort_c(A, p, r):
    # 递归终止条件
    if p >= r:
        return
    # 取 p 到 r 之间的中间位置 q
    q = p + (r - p) / 2
    # 分治递归
    merge_sort_c(A, p, q)
    merge_sort_c(A, q+1, r)
    # 将 A[p...q] 和 A[q+1...r] 合并为 A[p...r]
    merge(A[p...r], A[p...q], A[q+1...r])
    
# 将已经有序的两个子数组合并成一个新的有序数值的函数
def merge(A[p...r], A[p...q], A[q+1...r]):
    i, j, k = p, q+1, 0  # 初始化变量 i, j, k
    tmp = [0 for _ in range(len(A))]  # 申请一个大小跟 A[p...r] 一样的临时数组
    while (i <= q) and (j <= r):
        if A[i] <= A[j]:
            tmp[k] = A[i]
            i += 1
        else:
            tmp[k] = A[j]
            j += 1
        k += 1
        
    # 判断哪个子数组中有剩余的数据
    start, end = i, q
    if j <= r:
        start, end = j, r
    
    # 将剩余的数据拷贝到临时数组 tmp
    while start <= end:
        tmp[k] = A[start]
        k += 1
        start += 1
    
    # 将 tmp 中的数组拷贝回 A[p...r]
    for i in range(r-p):
        A[p+i] = tmp[i]
```

> 之后会思考一下，如何使用「哨兵」来简化 merge() 函数的编程

**性能分析**：

- 归并排序稳不稳定取决于 `merge()` 函数的实现；
- 时间复杂度：$\mathcal{O}(n\log n)$；
- 空间复杂度：$\mathcal{O}(n)$；

## 拓扑排序

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

## 堆排序

## BF/RK 字符串匹配算法

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

## 贪心算法（Greedy Algorithm）

贪心算法的解题步骤：

- **贪心算法适用的问题**：针对一组数据，定义了限制值和期望值，希望从中选出几个数据，在满足限制值的情况下，期望值最大；
- **初步尝试**：选择当前情况下，在对限制值同等贡献量的情况下，对期望值贡献最大的数据；
- **举例子**：检查贪心算法产生的结果是否最优。因为贪心贪心算法并不总能够给出最优解；

> 王争：掌握贪心算法的关键是**多练习**。不要刻意去记忆贪心算法的原理，多练习才是最有效的学习方法。

以下具体例子，应该可以帮助自己去理解贪心算法：

**1. 分糖果**：有 $m$ 个糖果和 $n$ 个孩子（孩子比糖果多（$m<n$），要把糖果分配给孩子。糖果有大小之分，并且每个孩子对于糖果大小的需求是不一样的，如何分配才能够尽可能地满足最多数量的孩子？

**2. 钱币找零**：纸币面值有 1、2、5、10、20、50、100 元的，每种面值的纸币有张数限制，如何用最少张的纸币去支付 $K$ 元？

> 用户「开心小毛」留言：
>
> 找零问题不能用贪婪算法，即使有面值为 1 元的币值也不行：考虑币值为 100，99 和 1 的币种，每种各 100 张，找 396 元。动态规划可求出 4 张 99 元，但贪心算法解出需 3 张 100 和 96 张 1 元。
>
> [LeetCode 322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

**3. 区间覆盖**：有 $n$ 个区间，起始端点和结束端点分别为 $[l_1,r_1], [l_2, r_2],\dots$，选出两两不相交的区间，最多可以选出多少个？

### 霍夫曼编码（Huffman Coding）

霍夫曼编码是一种十分有效的编码方法，广泛用于数据压缩中。压缩率通常在 20% ~ 90% 之间。

霍夫曼编码不仅会考察文本中有多少个不同字符，还会考察每个字符出现的频率。出现频率较多的字符，用稍微短一些的编码，出现频率少的，用稍微长一点的编码。

**问题**：假设有一个包含 1000 个字符的文件，每个字符占 1 个 byte（1 byte = 8 bits），存储这 1000 个字符就需要 8000 biits，如何对这些数据进行压缩？

假设文件中只有 6 个字符 a、b、c、d、e、f，并且出现频率从高到低排序。可以将其编码为

| 字符 | 出现频率 | 编码  | 总二进制位数 |
| :--: | :------: | :---: | :----------: |
|  a   |   450    |   1   |     450      |
|  b   |   350    |  01   |     700      |
|  c   |    90    |  001  |     270      |
|  d   |    60    | 0001  |     240      |
|  e   |    30    | 00001 |     150      |
|  f   |    20    | 00000 |     100      |

这样一来，任何字符的编码都不会是另一个的**前缀**，在解压缩的时候，读取尽可能长的可解压二进制串。经过这种编码之后，这 1000 个字符只需要 1910 bits 就可以了，压缩率 73.75%。

问题是，如何根据字符出现频率的不同，给不同的字符进行不同长度的编码，才能不混淆不同的字符？方法如下：

把每个字符看做一个节点，并且把频率也放到优先队列当中。从队列中取出频率最小的两个节点 A、B，然后新建一个节点 C，把频率设置为两个节点的频率之和，并把这个新节点 C 作为节点 A、B 的父节点。最后再把 C 节点放入到优先级队列中。重复这个过程，直到队列中没有数据。现在，我们给每一条边加上画一个权值，指向左子节点的边我们统统标记为 0，指向右子节点的边，我们统统标记为 1，那从根节点到叶节点的路径就是叶节点对应字符的霍夫曼编码。

王争老师的课程[《数据结构与算法之美》第 37 讲](http://gk.link/a/11bwH)中有一张图片，结合图片去看就很好理解了。

![huffman_coding_queue](https://static001.geekbang.org/resource/image/7b/7a/7b6a08e7df45eac66820b959c64f877a.jpg)

### Prim 和 Kruskal 最小生成树算法

### Dijkstra 单源最短路径算法

## 分治算法（Divide and Conquer）

> MapReduce、GFS 和 Bigtable 是 Google 大数据处理的三驾马车。

分治算法是一种处理问题的思想，递归是一种编程技巧，这就是分治和递归的区别所在。分治算法的核心思想：分而治之。分治算法的递归实现中，每一层递归都会涉及这三个操作：

- 分解：将原问题分解成一系列子问题；
- 解决：递归地求解各个子问题，直到子问题足够小到可以直接求解；
- 合并：将子问题的结果合并成原问题；

能够用分治算法解决的问题，需要满足：

- 原问题与分解成的小问题具有相同的模式；
- 原问题分解成的子问题可以独立求解，子问题之间没有相关性；
- 具有分解终止条件 —— 有足够小到可以直接求解的子问题；
- 子问题合并成原问题的操作复杂度不能太高；

分治算法的应用：

- 对于海量数据的情况（数据量远远大于内存的大小），可以用分治的思路，用多线程或者多机处理，加快处理速度；

**问题**：如何编程求出一组数据的有序对个数或者逆序对个数？

利用[归并排序](https://strongnine.github.io/9Docs/dev/leetcode/algorithm/#.-分治算法（Divide-and-Conquer）)算法，将两个有序的小数组，合并成一个有序的数组，同时计算逆序对个数。

> 王争老师的课程[《数据结构与算法之美》第 38 讲](http://gk.link/a/11bHy)有相关代码可供参考。
>

**MapReduce 的本质就是分治**：其框架只是一个人物调度器，底层依赖 GFS 来存储数据，依赖 Borg 管理机器。从 GFS 中拿数据，交给 Borg 中的机器执行，并且时刻监控机器执行的进度，一旦出现机器宕机、进度卡壳，就重新从 Borg 中调度一台机器执行。

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



## A* 算法

## 哈希算法

## 字符串匹配算法

字符串匹配算法中 BM、KMP、AC 自动机都是比较难懂的算法，对于算法有一定基础的人来说，想要看懂也不容易。要求是能够看懂就好，不要求自己能够实现。

### Trie 树

面试官喜欢考 Trie，但是要求能够看懂，会结合应用场景来考，考察的是面试者知道啥时候要用 Trie 树。

### BM

### KMP

### AC 自动机

## LeetCode 刷题总结

### 最高有效位

如果正整数 $y$ 是 2 的整数次幂，则 $y$ 的二进制表示中只有最高位为 1，其余都是 0，因此 $y\& (y-1)=0$. 如果 $y\le x$，则称 $y$ 为 $x$ 的「最高有效位」。

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用这样的一个方式，动态地维护最高有效位 `highBit`，然后算出所有小于给定整数 $n$ 的整数二进制表示包含 1 的数量。

### 最低有效位

对于正整数 $x$，将其二进制表示右移一位，等价于将其二进制表示的最低位去掉，得到的数是 $\lfloor\frac{x}{2}\rfloor$，如果 $bits[\lfloor\frac{x}{2}\rfloor]$ 的值已知，则可以得到 $bits[x]$ 的值：

- 如果 $x$ 是偶数，则 $bits[x]=bits[\lfloor\frac{x}{2}\rfloor]$；
- 如果 $x$ 是奇数，则 $bits[x] = bits[\lfloor\frac{x}{2}\rfloor]+1$.

上述两种情况可以合并成：$bits[x] $ 的值等于 $bits[\lfloor\frac{x}{2}\rfloor]$ 的值加上 $x$ 除以 2 的余数。

由于 $\lfloor\frac{x}{2}\rfloor$ 可以通过 $x\gg1$ 得到，$x$ 除以 2 的余数可以通过 $x \& 1$ 得到，因此有：$bits[x]=bits[x\gg1]+(x\&1)$.

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用最低有效位，遍历从 1 到 $n$ 的每个正整数 $i$，计算 $bits$ 的值，最终得到的数组 $bits$ 即为答案。

### Brian Kernighan 算法

记 $f(x)$ 表示 $x$ 和 $x-1$ 进行「与」运算所得的结果（即 $f(x)=x\&(x−1)$），那么 $f(x)$ 恰为 $x$ 删去其二进制表示中最右侧的 1 的结果。参考 LeetCode 题目 [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)。

例如 $x=10001000$，$x-1=10000111$，那么 $x\&(x-1)=10000000$. 

利用 Brian kernighan 算法计算出一个数的二进制表示有多少个 1 的方法如下。不断让 $s=f(s)$，直到 $s=0$。每循环一次 $s$ 都会删除二进制表示中最右侧的 1，最终的循环次数即为 $s$ 二进制表示中的 1 的数量。

## 索引

软件开发中，抛开业务和功能的外壳，本质都可以抽象为「对数据的存储和计算」，「存储」需要的就是数据结构，「计算」需要的就是算法。**索引**设计得好，可以节省存储空间、提高数据的处理效率。

索引的需求定义：

- 功能性需求：
  - **数据是格式化数据还是非格式化数据。**MySQL 中的是结构化数据，搜索引擎中的网页是非结构化数据；
  - **数据是静态数据还是动态数据。**静态数据不会有数据的增加、删除、更新操作；
  - **索引存储在内存还是硬盘。**数据在内存上的查询速度比在磁盘中的快；
  - **单值查找还是区间查找。**有一些数据结构不支持区间查找；
  - **单关键词查找还是多关键词组合查找。**可以通过集合操作（并交集）来得到多关键词查询；
- 非功能性需求
  - **无论是那种存储方式，索引对存储空间的消耗不能太大**；
  - **在考虑索引查询效率的同时，还要考虑索引的维护成本。**对原始数据的增删改时，也要维护索引；

构建索引常用的数据结构：

- 散列表：增删改查操作的性能好，具有 $\mathcal{O}(1)$ 的时间复杂度。例如 Redis、Memcache 等键值数据库；
- 红黑树：具有 $\mathcal{O}(\log n)$ 的时间复杂度。例如 Ext 文件系统中对磁盘块的索引；
- B+ 树： 

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

[1] [《数据结构与算法之美》王争｜极客时间](http://gk.link/a/11bwG)

[2] [LeetCode 官方题解](https://leetcode-cn.com/u/leetcode-solution/)

