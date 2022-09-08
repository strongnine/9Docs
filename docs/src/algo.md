## 复杂度

平均时间复杂度：全称加权平均时间复杂度或者期望时间复杂度。

均摊时间复杂度：适用于一个数据结构进行一组连续操作中，大部分情况下时间复杂度都很低，只有个别情况下时间复杂度比较高，而且这些操作之间存在前后连贯的时序关系。

## 排序

常用的排序算法复杂度对比

| 排序算法 |      平均时间复杂度      |         最好情况         |         最坏情况         |      空间复杂度       | 排序方式 | 稳定性 |
| :------: | :----------------------: | :----------------------: | :----------------------: | :-------------------: | :------: | :----: |
| 冒泡排序 |    $\mathcal{O}(n^2)$    |     $\mathcal{O}(n)$     |    $\mathcal{O}(n^2)$    |   $\mathcal{O}(1)$    |   原地   |  稳定  |
| 选择排序 |    $\mathcal{O}(n^2)$    |    $\mathcal{O}(n^2)$    |    $\mathcal{O}(n^2)$    |   $\mathcal{O}(1)$    |   原地   | 不稳定 |
| 插入排序 |    $\mathcal{O}(n^2)$    |     $\mathcal{O}(n)$     |    $\mathcal{O}(n^2)$    |   $\mathcal{O}(1)$    |   原地   |  稳定  |
| 希尔排序 | $\mathcal{O}(n^{1.25})$  |     $\mathcal{O}(n)$     | $\mathcal{O}(n\log^2 n)$ |   $\mathcal{O}(1)$    |   原地   | 不稳定 |
| 归并排序 |  $\mathcal{O}(n\log n)$  |  $\mathcal{O}(n\log n)$  |  $\mathcal{O}(n\log n)$  |   $\mathcal{O}(n)$    |  非原地  |  稳定  |
| 快速排序 |  $\mathcal{O}(n\log n)$  |  $\mathcal{O}(n\log n)$  |    $\mathcal{O}(n^2)$    | $\mathcal{O}(\log n)$ |   原地   | 不稳定 |
|  堆排序  |  $\mathcal{O}(n\log n)$  |  $\mathcal{O}(n\log n)$  |  $\mathcal{O}(n\log n)$  |   $\mathcal{O}(1)$    |   原地   | 不稳定 |
| 计数排序 |    $\mathcal{O}(n+k)$    |    $\mathcal{O}(n+k)$    |    $\mathcal{O}(n+k)$    |   $\mathcal{O}(k)$    |  非原地  |  稳定  |
|  桶排序  |    $\mathcal{O}(n+k)$    |    $\mathcal{O}(n+k)$    |     $\mathcal{O}(n)$     |  $\mathcal{O}(n+k)$   |  非原地  | 不稳定 |
| 基数排序 | $\mathcal{O}(n\times k)$ | $\mathcal{O}(n\times k)$ | $\mathcal{O}(n\times k)$ |  $\mathcal{O}(n+k)$   |  非原地  |  稳定  |

## 动态规划

### 背包问题（含物品价值）

```python
def knapsack3(weight: List[int], value: List[int], n: int, w: int) -> int:
    # states[i][j] 代表决策完前 i 个物品时背包重量为 j 时的所有物品价值只和的最大值
    states = [[-1 for _ in range(w + 1)] for _ in range(n)]
    
    states[0][0] = 0
    if weight[0] <= w: 
        states[0][weight[0]] = value[0]
    
    for i in range(1, n):
        for j in range(w + 1):  # 不选择第 i 个物品
            if states[i - 1][j] >= 0: 
                states[i][j] = states[i - 1][j]
        
        for j in range(w - weight[i] + 1):  # 选择第 i 个物品
            if states[i - 1][j] >= 0:
                v = states[i - 1][j] + value[i]
                if v > states[i][j + weight[i]]:
                    states[i][j + weight[i]] = v
                    
    # 找出最大值
    maxvalue = -1
    for j in range(w + 1):
        if states[n - 1][j] > maxvalue:
            maxvalue = states[n - 1][j]

    return maxvalue
```



