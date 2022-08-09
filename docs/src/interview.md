## 笔试

### 面试：输入输出的处理

Python 的输出输出处理：

```python
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
num = int(input().strip())  # 输入一个整数
nums = list(map(int, input().strip().split()))  # 输入一个整数列表
```

### 面试题 1：整数除法

**题目**：输入 2 个 int 型整数，它们进行除法计算并返回商，要求不得使用乘号 *、除号 / 及求余符号 %。当发生溢出时，返回最大的整数值。假设除数不为 0。例如，输入 15 和 2，输出 15/2 的结果，即 7。

- 用减法实现除法，时间复杂度 $\mathcal{O}(\log n)$
- 对于含有负数的情况，可以先记录最终答案的是正是负，然后全部转化为绝对值计算，最后在返回时操作；
- 由于是整数的除法，且除数不等于 0，因此商的绝对值一定小于等于被除数的绝对值。只有一种情况会导致溢出，即 $(-2^{31})/(-1)$。

```python
import sys
def divide(dividend: int, divisor: int) -> int:
    # 0x80000000 为最小的 int 整数，即 -2^31
    if (dividend == 0x80000000 and divisor == -1):
        return sys.maxint  # 溢出时返回最大整数
    
    # 如果 negative == 1 则结果为 负数，否则为 正数
    negative = 2
    if dividend > 0:
        negative -= 1
        dividend = -dividend
        
    if divisor > 0:
        negative -= 1
        divisor = -divisor
    
    result = divideCore(dividend, divisor)
    return -result if negative == 1 else result

def divideCore(dividend: int, divisor: int) -> int:
    """
    使用减法实现两个负数的除法
    """
    result = 0
    while dividend <= divisor:
        value = divisor
        quotient = 1  # 商
        while (value >= 0xc0000000) and (dividend <= value + value):
            # 0xc0000000 是 0x80000000 的一半，即 -2^30（防止溢出）
            quotient += quotient
            value += value
            
        result += quotient
        dividend -= value
    
    return result
```

### 面试题 2：二进制加法

**题目**：输入两个表示二进制的字符串，请计算它们的和，并以二进制字符串的形式输出。例如，输入的二进制字符串分别是「11」和「10」，则输出「101」。

```python
def binaryAdd(a, b):
    ans = ""
    i = len(a) - 1
    j = len(b) - 1
    carry = 0
    while (i >= 0) or (j >= 0):
        digitA = a[i] - '0' if i >= 0 else 0
        digitB = b[j] - '0' if j >= 0 else 0
        i += 1
        j += 1
        sum_ = digitA + digitB + carry
        carry = sum_ - 2 if sum_ >= 2 else sum_
        ans.append(str(sum_))
    
    if carry == 1:
        ans.append("1")
    
    # 字符串翻转的两种方法
    return ans[::-1]  # 使用字符串切片
	# return ''.join(reversed(ans))  # 使用 reversed()
```

### 面试题 3：前 n 个数字二进制形式中 1 的个数

**题目**：输入一个非负数 n，请计算 0 到 n 之间每个数字的二进制形式中 1 的个数，并输出一个数组。例如，输入的 n 为 4，由于 0、1、2、3、4 的二进制形式中 1 的个数分别为 0、1、1、2、1，因此输出数组 `[0, 1, 1, 2, 1]`。

#### 计算每个整数的二进制形式中 1 的个数

```python
def countBits(num):
    result = [0 for _ in range(num + 1)]
    for i in range(num + 1):
        j = i
        while j != 0:
            result[i] += 1
            # x & (x - 1) 的操作可以将整数 x 二进制的最右边的 1 变成 0
            j = j & (j - 1)
            
    return result
```

上述代码的时间复杂度 $\mathcal{O}(nk)$，$k$ 为二进制中 1 的个数

#### 根据 `x & (x - 1)` 计算其二进制形式中 1 的个数

整数 `x` 的二进制形式中 1 的个数比 `x & (x - 1)` 的多 1 个

```python
def countBits(num):
    result = [0 for _ in range(num + 1)]
    for i in range(num + 1):
        result[i] = result[i & (i + 1)] + 1
        
    return result
```

这个代码的时间复杂度 $\mathcal{O}(n)$. 

#### 根据 `x / 2` 计算 `x` 的二进制形式中 1 的个数

如果正整数 `x` 是一个偶数，那么 `x` 相当于将 `x / 2` 左移一位的结果，他们两个的二进制中的 1 个数是相同的。如果 `x` 是奇数，那么 `x` 相当于将 `x / 2` 左移一位之后再将最右边一位设为 1 的结果，因此奇数 `x` 的二进制形式中 1 的个数比 `x / 2` 的个数多 1 个。

```python
def countBits(num):
    result = [0 for _ in range(num + 1)]
    for i in range(num + 1):
        # 位运算效率更高：
        # 用 i >> 1 替代 i / 2
        # 用 i & 1 替代 i % 2
        result[i] = result[i >> 1] + (i & 1)
    
    return result
```

这种解法时间复杂度 $\mathcal{O}(n)$

### 面试题 4：只出现一次的数字

题目：输入一个整数数组，数组中只有一个数字出现了一次，而其他数字都出现了 3 次。请找出那个只出现一次的数字。例如，如果输入的数组为 `[0, 1, 0, 1, 0, 1, 100]`，则只出现一次的数字是 100。

> 简单版本：输入数组中除一个数字只出现一次之外其他数字都出现两次，请找出只出现一次的数字。因为任何一个数字异或它自己的结果都是 0，因此解法就是将数组中所有数字进行异或运算，最终的结果就是那个只出现一次的数字。

这个题目与简单版本的不同就是，其他重复的数字是出现 3 次的。

思路是将数组中所有数字的同一位置的数位相加，得到的结果中每个数位都除以 3，如果能够整除，那么只出现一次的数字，对应的数位就是 0，如果结果余 1，那么对应的数位就是 1.

```python
def singleNumber(nums: List[int]):
    bitSums = [0 for _ in range(32)]  # 一个整数是由 32 个 0 或 1 组成的
    for num in nums:
        for i in range(32):
            bitSums[i] += (num >> (31 - i)) & 1  # 得到整数 num 的二进制形式中左数起第 i 个数位
            
    result = 0
    for i in range(32):
        result = (result << 1) + bitSums[i] % 3
    
    return result
```

进阶题目：输入一个整数数组，数组中只有一个数字出现 m 次，其他数字都出现 n 次。请找出那个唯一出现 m 次的数字。假设 m 不能被 n 整除。

如果数组中所有数字的第 i 个数位相加之和能被 n 整除，那么出现 m 次的数字的第 i 个数位一定是 0；否则出现 m 次的数字的第 i 个数位一定是 1

### 面试题 5：单词长度的最大乘积

**题目**：输入一个字符串数组 words，请计算不包含相同字符的两个字符串 words[i] 和 words[j] 的长度乘积的最大值。如果所有字符串都包含至少一个相同字符，那么返回 0。假设字符串中只包含英文小写字母。例如，输入的字符串数组 words 为 ["abcw"，"foo"，"bar"，"fxyz"，"abcdef"]，数组中的字符串 "bar" 与 "foo" 没有相同的字符，它们长度的乘积为 9。"abcw" 与 "fxyz" 也没有相同的字符，它们长度的乘积为 16，这是该数组不包含相同字符的一对字符串的长度乘积的最大值。

**暴力法**：对于 str1 中的每个字符 ch，扫描字符串 str2 判断字符 ch 是否出现在 str2 中。如果两个字符串的长度分别为 p 和 q，那么暴力法时间复杂度 $\mathcal{O}(pq)$.

**哈希表记录字符串中出现的字符**：题目假设字符串只包含英文小写字母，用长度为 26 的数组模拟哈希表

```python
def maxProduct(words: List[str]):
    n = len(words)
    flags = [[0 for _ in range(26)] for _ in range(len(words))]
    # Step 1. 初始化每个字符串对应的哈希表
    for i in range(n):
        word = words[i]
        for ch in word:
            flags[i][ord(ch) - ord('a')] = 1
    
    result = 0
    for i in range(n):
        # Step 2. 根据哈希表判断每对字符串是否包含相同的字符
        for j in range(i + 1, n):
            k = 0
            while k < 26:
                if flags[i][k] and flags[j][k]:
                    break
                k += 1
            
            # Step 3. 如果所有字符都不相同，计算乘积
            if k == 26:
                prod = len(words[i]) * len(words[j])
                result = max(result, prod)
    
    return result
```

第 1 步如果 words 的长度为 n，平均每个字符串长度为 k，那么时间复杂度为 $\mathcal{O}(nk)$；第 2 步总共有 $n^2$ 对字符串，时间复杂度为 $\mathcal{O}(n^2)$，总的时间复杂度为 $\mathcal{O}(nk+n^2)$.

**用整数的二进制数位记录字符串中出现的字符**：int 型整数的有 32 个数位，将二进制的最右边代表 ‘a’，倒数最右代表 ‘b’，对应的数位为 1 则代表 words[i] 包含该字母，否则不包含。如果两个字符串没有相同的字符，那么它们对应的整数的「与」运算结果等于 0.

```python
def maxProduct(words: List[str]):
    n = len(words)
    flags = [0 for _ in range(n)]
    for i in range(n):
        word = word[i]
        for ch in word:
            flag[i] |= 1 << (ch - 'a')
            
    result = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (flags[i] & flags[j] == 0):
                prod = len(words[i]) * len(words[j])
                result = max(result, prod)
    
    return result
```

这种解法的时间复杂度也是 $\mathcal{O}(nk+n^2)$，空间复杂度 $\mathcal{O}(n)$，但是这种解法在判断两个字符串是否包含相同字符时只需要 1 次运算，而前面的需要 26 次。

### 面试题 6：排序数组中的两个数字之和

题目：输入一个**递增排序**的数组和一个值 k，请问如何在数组中找出两个和为 k 的数字并返回它们的下标？假设数组中存在且只存在一对符合条件的数字，同时一个数字不能使用两次。例如，输入数组 [1，2，4，6，10]，k 的值为 8，数组中的数字 2 与 6 的和为 8，它们的下标分别为 1 与 3。

**双指针**：初始下指针 p1 指向数组下标 0，指针 p2 指向数组末尾，如果两个指针指向的数字之和小于 k，可以把指针 p1 向右移动增加和的大小；如果两个指针指向的数字之和大于 k，可以把指针 p2 向左移动减小和的大小；如果两个指针指向的数字之和等于 k，那么就找到了符合条件的两个数字。

```python
def twoSum(numbers: int, target: int):
    i = 0
    j = len(numbers) - 1
    while i < j and numbers[i] + numbers[j] != target:
        if numbers[i] + numbers[j] < target:
            i += 1
        else:
            j -= 1
    
    return [i, j]
```

### 面试题 7：数组中和为 0 的 3 个数字

题目：输入一个数组，如何找出数组中所有和为 0 的 3 个数字的三元组？需要注意的是，返回值中不得包含重复的三元组。例如，在数组 [-1，0，1，2，-1，-4] 中有两个三元组的和为 0，它们分别是 [-1，0，1] 和 [-1，-1，2]。

先对数组进行排序，在固定用变量 i 指向的数字之后，用函数 twoSum 在排序后的数组中找出所有下标大于 i 并且和为 -nums[i] 的两个数字（下标分别为 j 和 k）。如果 nums[i], nums[j], nums[k] 的和大于 0，那么下标 k 向左移动；如果 nums[i], nums[j], nums[k] 的和小于 0，那么下标 j 向右移动；如果 3 个数字之和正好等于 0，那么向右移动下标 j，以便找到其他和为 -nums[i] 的两个数字。

```python
def threeSum(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    result = []
    if n >= 3:
        nums.sort()
        i = 0
        while (i < n - 2):
            twoSum(nums, i, result)
            temp = nums[i]
            while(i < n and nums[i] == temp):
                i += 1
    return result

def twoSum(nums: List[int], i: int, result: List[List[int]]) -> None:
    n = len(nums)
    j = i + 1
    k = n - 1
    while j < k:
        if nums[i] + nums[j] + nums[k] == 0:
            result.append([i, j, k])
            
            temp = nums[j]
            while (nums[j] == temp and j < k):
                j += 1
            elif nums[i] + nums[j] + nums[k] < 0:
                j += 1
            else:
                k -= 1
```

### 面试题 8：和大于或等于 k 的最短子数组

**题目**：输入一个**正整数**组成的数组和一个正整数 k，请问数组中和大于或等于 k 的连续子数组的最短长度是多少？如果不存在所有数字之和大于或等于 k 的子数组，则返回 0。例如，输入数组 [5，1，4，3]，k 的值为 7，和大于或等于 7 的最短连续子数组是 [4，3]，因此输出它的长度 2。

> 子数组由数组中一个或连续的多个数字组成。

指针 p1 和 p2 初始指向数组的第 1 个元素，由于数组中的数字都是正整数：

- 如果两个指针之间的子数组中所有数字之和大于等于 k，那么 p1 向右移动，相当于子数组删除最左边的元素；
- 如果两个指针之间的子数组中所有数字之和小于 k，那么 p2 向右移动，相当于子数组在最右边添加一个数字；

```python
def minSubArrayLen(k: int, nums: List[int]) -> int:
    left = 0
    sum_ = 0 
    minLength = float('inf')
    for right in range(len(nums)):
        sum_ += nums[right]
        while left <= right and sum_ >= k:
            minLength = min(minLength, right - left + 1)
            sum_ -= nums[left]
            left += 1
    
    return minLength if minLength < float('inf') else 0
```

时间复杂度：$\mathcal{O}(n)$

### 面试题 9：乘积小于 k 的子数组

**题目**：输入一个由**正整数**组成的数组和一个正整数 k，请问数组中有多少个数字乘积小于 k 的连续子数组？例如，输入数组 [10，5，2，6]，k 的值为 100，有 8 个子数组的所有数字的乘积小于 100，它们分别是 [10]、[5]、[2]、[6]、[10，5]、[5，2]、[2，6] 和 [5，2，6]。

用指针 p1 和 p2 指向数组中的两个数字，初始时都指向数组的第一个元素，指针 p1 永远不会走到指针 p2 的右边：

- 如果两个指针之间的子数组中数字的乘积小于 k，指针 p2 向右移动；
- 如果两个指针之间的子数组中数字的乘积大于等于 k，指针 p1 向右移动；

**找出所有数字乘积小于 k 的子数组的个数**：一旦 p1 向右移动到某个位置时子数组的乘积小于 k，只需要保持 p2 不懂，向右移动 p1 形成的所有子数组的数字乘积就一定小于 k

```python
def numSubarrayProductLessThanK(nums: List[int], k: int) -> int:
    product = 1
    left = 0
    count = 0
    for right in range(len(nums)):
        product *= nums[right]
        while left <= right and product >= k:
            product /= nums[left]
            left += 1
    
        count += right - left + 1 if right >= left else 0
    
    return count
```

### 面试题 10：和为 k 的子数组

**题目**：输入一个整数数组和一个整数 k，请问数组中有多少个数字之和等于 k 的连续子数组？例如，输入数组 [1，1，1]，k 的值为 2，有 2 个连续子数组之和等于 2。

**注意**：该问题与前面两个问题的区别在于，数组中的数字不是**正整数**，如果使用双指针的方式，无法保证指针右移，数组中的和就变大（或变小）。

```python
from collections import defaultdict
def subarraySum(nums: List[int], k; int) -> int:
    sumToCount = defaultdict(int)  # 哈希表保存从第 1 个数到当前扫描到的数字之间的数字之和
    sumToCount[0] = 1
    
    sum_ = 0
    count = 0
    for num in nums:
        sum_ += num
        count += sumToCount[sum_ - k]
        sumToCount[sum_] = sumToCount[sum_] + 1
    
    return count
```

时间复杂度：$\mathcal{O}(n)$；空间复杂度：$\mathcal{O}(n)$

