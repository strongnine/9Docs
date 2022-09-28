## 笔试

### 笔试：输入输出的处理

Python 的输出输出处理：

在 Python 中调用 `print` 时，实际上是调用 `sys.stdout.write(obj + '\n')`。`sys.stdin.readline()` 会将标准输入全部获取，包括末尾的换行符 `\n`，因此用 `len` 计算长度是包含了换行符的，用这种方法输入时可以用 `strip()` 来去掉换行符（或者用 `sys.stdin.readline()[:-1]`）。

`sys.stdin.readline().trip()` 就等价于 `input()`。

> 注意：Python 2 和 Python 3 的输入输出稍微有区别。在使用 Python 2 的时候最好用 `sys.stdin` 的方式读取，用 `input()` 对于字符串和数字相混合的输入会报错。
>
> 例如对于 `S 0` 这种数据，Python 2 要求对于字符串需要用引号，如 `“S” 0`，否则无法识别。

```python
import sys
# strip() 去掉两端的空白符，返回 str
# split() 按照空白符分割，返回 List[str]
# map(type, x) 把列表 x 中的元素映射成类型 type
# 1. 题目没有告知多少组数据时，用 while True
while True:
    try:
        # ...
    except:
        break
        
# 2. 题目告知有 T 组数据时，用 For loop
T = int(input().strip())
for _ in range(T):
    # ...
    
    
# 3. 不同的输入
s = input().strip()  # 输入一个字符
str_ = sys.stdin.readline().strip()  # 读取一行
num = int(input().strip())  # 输入一个整数
nums = list(map(int, input().strip().split()))  # 输入一个整数列表
```



### 通过排除法找质数个数

给定以下代码一个数，返回该数范围内的素数个数。

```python
import math
def sieve(size):
    sieve = [True] * size
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(math.sqrt(size)) + 1):
        k = i * 2
        while k < size:
            sieve[k] = False
            k += i
    return sum(1 for x in sieve if x)
```

> 质数：又称素数，在大于 1 的自然数中，除了 1 和该数自身之外，无法被其他自然数整除的数，即只有 1 于其本身两个正因数。





### 下一个字典序

给你一个整数数组 `nums` ，找出 `nums` 在字典序中的下一个排列。必须 `原地` 修改，只允许使用额外常数空间。

需要找到一个左边的「较小数」和右边的「较大数」，同时要求「较小数」尽量靠右，「较大数」尽可能小。具体做法如下：

- 从后向前查找第一个顺序对 `(i, i + 1)`，满足 `a[i] < a[i + 1]`，`a[i]` 即为想要的「较小数」，此时 `(i + 1, n)` 必然是下降序列；
- 在区间 `[i + 1, n - 1]` 中从后向前查找第一个元素 `j` 满足 `a[i] < a[j]`，`a[j]` 即为想要找的「较大数」；
- 交换 `a[i]` 与 `a[j]`，可以证明区间 `[i + 1, n - 1]` 必为降序，我们可以直接使用双指针反转区间 `[i + 1, n - 1]` 使其变为升序；

如果步骤 1 找不到满足条件的顺序对，说明当前序列已经是一个降序序列，是最大的序列，可以直接跳过步骤 2 执行步骤 3。

```python
def nextPermutation(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    i = len(nums) - 2
    # Step 1. 找到满足 a[i] < a[i + 1] 的顺序对
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:  # 找不到顺序对，说明当前序列已经是一个降序序列
        j = len(nums) - 1
        # Step 2. 在区间 [i + 1, n) 中查找第一个满足 a[i] < a[j] 的 j
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        # 交换 a[i], a[j], 可以证明区间 [i + 1, n) 一定是降序的
        nums[i], nums[j] = nums[j], nums[i]
        
    # Step 3. 用双指针的方式去反转区间 [i + 1, n) 使其变为升序
    left, right = i + 1, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
```



### 双十一购物（动态规划）

这个题目的基础是 0 - 1 背包：

```python
def double11advance(items: List[int], n: int, w: int, th: int) -> None:
    # w 是预算, th 是满减的门槛
    states = [[False for _ in range(w + 1)] for _ in range(n)]
    states[0][0]  # True 不买立省 100%
    if items[0] <= w:
        states[0][items[0]] = True
    
    for i in range(1, n):
        for j in range(w + 1):  # 不买第 i 个物品
            if states[i - 1][j] == True:
                states[i][j] = states[i - 1][j]
        
        for j in range(w - items[i] + 1):  # 购买第 i 个物品
            if states[i - 1][j] = True:
                states[i][j + items[i]] = True
	
    # 输出花费超过满减门槛又尽可能小的价格
    j = th
    while j < w:
        if states[n - 1][j] == True:
            print("薅羊毛最少花费： ", j)
            break
    	j += 1
    if j == w:
        print("选的这些东西薅不到羊毛呀！")
    else:
        print("可以薅到羊毛, 下面输出可以购买的物品清单.")
        for i in range(n - 1, 0, -1):
            if j - items[i] >= 0 and states[i - 1][j - items[i]] == True:
                print(i, end=" ")
                j = j - items[i]
		if j != 0:
            print(items[0], end=" ")
        print("\n")
```

关于输出可以购买的物品清单，状态 `[i][j]` 只有可能从 `[i - 1][j]` 或者 `[i - 1][j - value[i]]` 两个状态推导过来。因此可以检查这两个状态是否可达，即 states 里面是否为 True。假如两个状态都为 True，那么就随便选择一个。
