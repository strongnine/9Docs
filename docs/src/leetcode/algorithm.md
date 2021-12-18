# 算法

## 最高有效位

如果正整数 $y$ 是 2 的整数次幂，则 $y$ 的二进制表示中只有最高位为 1，其余都是 0，因此 $y\& (y-1)=0$. 如果 $y\le x$，则称 $y$ 为 $x$ 的「最高有效位」。

题目 [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/) 利用这样的一个方式，动态地维护最高有效位 `highBit`，然后算出所有小于给定整数 $n$ 的整数二进制表示包含 1 的数量。

## 最低有效位

对于正整数 $x$，将其二进制表示右移一位，得到

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

## 回溯算法（Backtrack）

适用问题：

- 解决一个问题有多个步骤
- 每个步骤有多种方法
- 需要找出所有的方法

原理：在一棵树上的**深度优先遍历**
