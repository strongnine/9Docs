## 个人心得

这个笔记是本人在跟着 [labuladong 的算法小抄](https://labuladong.github.io/algo/)进行学习时候的个人笔记，主要是记录一些重要的点，算法的练习无论如何还是在于自己真正地上手去练习。笔记中的代码，也是根据「算法小抄」中提供的代码框架，改成 Python 的版本，作为一个记录，方便自己在刷题的时候有个地方可以对照着。

我想这些框架，就像是上学的时候写数学题的公式，把公式整理到笔记本上， 在做题的时候，就直接代公式。只是重要的是要学习如何代公式，这个部分是自己的思考，逃不开练习。

笔记的最后有列出参考资料，一些句子不是自己写的，会用「引用」并且「上标」的方式来注明。

## 数据结构

在最底层中，数据结构的存储方式只有两种方式：数组（顺序存储）和链表（链式存储）。而其他的例如散列表、栈、队列、堆、树、图等等的数据结构，都是在数组和链表上的特殊操作。

而对于任何数据结构，基本操作就是「遍历」和「访问」，也就是「增删查改」。线性的形式以 `for/while` 迭代为代表，非线性的形式以递归为代表。

> 算法的本质就是「穷举」，但穷举有两个关键难点：无遗漏、无冗余。$^1$

## 二叉树

几乎所有二叉树的题目都是一套这个框架就出来了：

```python
def traverse(root):
    # 前序遍历代码位置
    traverse(root.left)
    # 中序遍历代码位置
    traverse(root.right)
    # 后序遍历代码位置
```

> 快速排序就是二叉树的前序遍历，归并排序就是二叉树的后序遍历。$^1$

快速排序的代码框架：

```python
def sort(nums, lo, hi):
    #===== 前序遍历位置 =====#
    # 通过交换元素构建分界点 p
    p = partition(nums, lo, hi)
    #======================#
    sort(nums, lo, p-1)
    sort(nums, p+1, hi)
```

归并排序的代码框架：

```python
def sort(nums, lo, hi):
    mid = lo + (hi - lo) / 2
    sort(nums, lo, mid)    # 排序 nums[lo...mid]
    sort(nums, mid+1, hi)  # 排序 nums[mid+1...hi]
    
    #===== 后序遍历位置 =====#
    # 合并 nums[lo...mid] 和 nums[mid+1...hi]
    merge(nums, lo, mid, hi)
    #======================#
```

层序遍历属于迭代遍历，代码框架：

```python
from collections import deque
# 输入一棵二叉树的根节点，层序遍历这棵二叉树
def levelTraverse(root):
    if (root == None): 
        return
    q = deque()
    q.append(root)
    
    # 从上到下遍历二叉树的每一层
    while q:
        sz = len(q)
        for i in range(sz):
            cur = q.popleft()
            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)
```

**笔记**$^1$：

- 只要涉及递归的问题，所有回溯、动态规划、分治算法，都是树的问题；
- 写树相关的算法，就是先搞清楚当前 `root` 节点「该做什么」以及「什么时候做」，然后根据函数定义递归调用子节点；

- 二叉树题目的「递归」解法可以分成两类思路：

  - 第一类：遍历一遍二叉树得出答案，对应的是「回溯算法」；

  - 第二类：通过分解问题计算出答案，对应的是「动态规划」；

- 首先思考是否可以通过遍历一遍二叉树得到答案？不能的话能够定义一个递归函数，通过子问题（子树）的答案推导出原问题的答案？

- 遇到子树问题，首先想到给函数设置返回值，在后序位置做文章；

## 二叉搜索树（Binary Search Tree, BST）

BST 代码框架：

```python
def BST(root, target):
    if (root.val == target):
        # 找到目标，增删改
    if (root.val < target):
        BST(root.right, target)  # 目标值大于当前节点值，在右边找
    if (root.val > target):
        BST(root.left, target)   # 目标值小于当前节点值，在左边找
```

BST 删除节点的代码：

```python
def delNode(root, key):
    if (root == None): return None
    if (root.val == key):
        # case 1 and 2
        if (root.left == None): return root.right
        if (root.right == None): return root.left
        # case 3
        minNode = getMin(root.right)
        # 这一步删除右子树最小结点，只会碰到 case 1：没有子节点
        root.right = delNode(root.right, minNode.val)
        minNode.left = root.left
        minNode.right = root.right
        return minNode
    elif (root.val > key):
        root.left = delNode(root.left, key)
    elif (root.val < key):
        root.right = delNode(root.right, key)
    return root

def getMin(node):
    while node.left: node = node.left
    return node
```

验证 BST 代码：

```python
def isValidBST(root):
    return isValid(root, None, None)

# 让 root 的信息能够传给子树节点
def isValid(root, minNode, maxNode):
    if (root == None): return True
    if (minNode and root.val <= minNode.val): return False
    if (maxNode and root.val >= maxNode.val): return False
    # 左子树最大的值不能超过 root 节点
    # 右子树最小的值不能超过 root 节点
    return isValid(root.left, minNode, root) and isValid(root.right, root, maxNode)
```



**笔记**$^1$：

- BST 特性：对于每一个结点 `node`（1）左子树节点的值都比 `node` 的要小，右子树节点的值都比 `node` 的大；（2）左右子树都是 BST；
- BST 的中序遍历结果是升序的（将遍历的顺序变成先遍历右结点再遍历左结点，结果就是降序的）；
- BST 相关的问题，要么利用 BST 左小右大的特性提升算法效率，要么利用**中序遍历**的特性满足题目的要求；
- BST 中删除节点 `node` 的三种情况：
  - `node` 是叶节点，直接变为 `None`；
  - `node` 只有一个非空子节点，直接让其替代自身；
  - 有两个非空子节点，用左子树最大的节点，或者右子树最小的节点接替自己；
- 一般最好不要通过修改节点内部值来实现交换节点，而是把整个节点进行交换；

## 回溯算法

> 回溯算法就是个 N 叉树的前后序遍历问题，没有例外。$^1$

回溯算法的特点：简单粗暴效率低，但又特别有用。

## 动态规划

## 贪心算法

> 所谓贪心算法就是在题目中发现一些规律（专业点叫贪心选择性质），使得你不用完整穷举所有解就可以得出答案。$^1$

动态规划问题的一般形式就是求最值，核心问题就是穷举，特殊之处在于存在「重叠子问题」，需要「备忘录」或者「DP table」来优化穷举过程。

## 参考

### 相关资料

[1]：[《数据结构与算法之美》王争｜极客时间](http://gk.link/a/11bwG)；

[2]：[LeetCode 官方题解](https://leetcode-cn.com/u/leetcode-solution/)；

[3]：[C 语言中文网](http://c.biancheng.net/)；

[4]：[CodeTop 企业题库](https://codetop.cc/home)：能够根据不同的公司选择题库，有专门性地刷；

[5]：[labuladong 的算法小抄](https://labuladong.github.io/algo/)：十分推荐的刷题资料，索引做得十分好；

[6]：[labuladong 刷题三件套](https://pan.baidu.com/s/1PoG0Zxy7H64aXUM-Gj0UuA)：《算法秘籍》、《刷题笔记》、刷题插件（提取码：541i）；


### 语录参考

1：labuladong

2：王争

3：强劲九

