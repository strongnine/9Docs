# 数据结构

各种数据类型的范围：

- `int`: $[-2147483648, 2147483647]$，大约等于 $-2\times 10^{9}\sim 2\times 10^{9}$.

位运算符：

| 运算符 |描述 | 实例 |
| :------: | :----: | :----: |
| &      | 按位「与」运算 |      |
| \|     | 按位「或」运算 |      |
| ^      | 按位「异或」运算 |      |
| ~      | 按位「取反」运算 |      |
| <<     | 二进制左移运算 |      |
| >>    | 二进制右移运算 |      |

## 向量（Vector）

```cpp
// 给向量 x 分配 size 个 value 值
x.assign(int size, int value);	
// 给向量 x 分配从迭代器初始位置到最终位置的值
x.assign(InputIterator first, InputIterator last);	

x.push_back();    // 添加元素至向量末端
x.pop_back();     // 将最后的元素弹出
```

## 栈（Stack）

```c++
using namespace std;
stack<int> stk;    // 声明一个栈 stk
stk.push(1);       // push 元素进栈
stk.pop();         // 弹出栈顶元素
skt.top();         // 查看栈顶元素
stk.empty();       // 判断栈是否为空
```



## 队列（Queue）

```c++
using namespace std;
// 生成方式
queue<string> q;
queue<string> q {arr};

q.front();            // 返回第一个元素的引用，如果 q 为空，返回值是未定义的
q.back();             // 返回最后一个元素的引用
q.push(const T& obj); // 在尾部添加一个元素的副本
q.emplace();          // 在尾部生成对象
q.pop();              // 删除第一个元素
q.size();             // 返回元素个数
q.empty();            // 判断是否为空
```

## 哈希表（Hash-map）

哈希字典：

```c++
using namespace std;
unordered_map<int, int> hash;  // 创建哈希表 hash
hash[k] = v;                   // 插入元素
hash.erase(k);                 // 移除元素
hash.clear();                  // 清空元素
// 打印 hash 表中所有元素
for (const auto& elemt : hash) {
    cout << "key: " << elemt.first << ", value: " << elemt.second << "\n";
}
```

哈希集合：

```c++
unordered_set<int> hash_set;
// 判断元素 x 是否在集合中
if (hash_set.find(x) != hash_set.end()) {
    return true;
}
```

## 树（Tree）

树的每一个元素叫做「节点」，用来连接相邻节点之间的关系，叫做「父子关系」。具有相同父节点的节点称为「兄弟节点」，没有节点的节点叫做「根节点」，没有子节点的节点叫做「叶子节点」或者「叶节点」。

- 高度（Height）：节点到叶子节点的最长路径（边数）；
- 深度（Depth）：根节点到这个节点所经历的边的个数；
- 层数（Level）：节点的深度 + 1；
- 数的高度：根节点的高度；

**二叉树（Binary Tree）**

最常用的树结构是二叉树。二叉事的每个节点最多有两个子节点，分别称为「左子节点」和「右子节点」。

**满二叉树**：叶子节点全部在最底层，除了叶子节点之外，每个节点都有左右两个子节点。

**完全二叉树**：叶子节点全都在最底下两层，最后一层的叶子节点都靠左排列，并且除了最后一层，其他层的节点个数都要达到最大。

存储二叉树的两种方法：

- 基于指针或者引用的**二叉链式存储法**。这种存储方法比较常用；
- 基于数组的**顺序存储法**。根节点存储在 `i = 1` 的位置，左子节点存储在 `2 * i = 2` 的位置，右子节点存储在 `2 * i + 1 = 3` 的位置，以此类推。下标 `i/2` 的位置存储的就是父节点；

一个完全二叉树用数组使用顺序存储法是最省内存的。

[二叉树的遍历](https://strongnine.github.io/9Docs/dev/leetcode/algorithm/#%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E9%81%8D%E5%8E%86)：

- **前序遍历**：对于树中的任意节点来说，先打印这个节点，再打印它的左子树，最后打印它的右子树。**节点 => 左子树 => 右子树**；
- **中序遍历**：对于树中的任意节点来说，先打印它的左子树，然后再打印它本身，最后打印它的右子树。**左子树 => 节点 => 右子树**；
- **后序遍历**：对于树中的任意节点来说，先打印它的左子树，然后再打印它的右子树，最后打印这个节点本书。**左子树 => 右子树 => 节点**；

**二叉查找树（Binary Search Tree）**

又名二叉搜索树，是为了实现快速查找而生的。不仅仅支持快速查找一个数据就，还支持快速插入、删除一个数据。

二叉查找树要求，在书中的任意一个节点，其左子树中的每一个节点的值，都要小于这个节点的值，而右子树节点的值都大于这个节点的值。

**二叉查找树的查找操作**：

- 如果根节点等于我们要查找的数，直接返回；
- 比如要查找的数据比根节点小，在左子树中递归查找；
- 否者如果比根节点大，在右子树中递归查找；

```c++
class BinarySearchTree {
private: 
    TreeNode tree;
    
public: 
    TreeNode find(int data) {
        TreeNode* p = tree;
        while (p != nullptr) {
            if (data < p->val) {p = p->left;}
            else if (data > p->val) {p = p->right;}
            else {return p;}
        }
        return null;
    }
}
```

**二叉查找树的插入操作**：

- 如果要插入的数据比节点的数据大：
  - 并且节点的右子树为空，新数据插到右子节点的位置；
  - 如果不为空，就再递归遍历右子树，查找插入位置；
- 如果要插入的数据比节点数值小：
  - 并且节点的左子树为空，新数据插到左子节点的位置；
  - 如果不为空，就再递归遍历左子树，查找插入位置；

```c++
void insert(int data) {
    if (tree == nullptr) {
        tree = new TreeNode(data);
        return;
    }
    
    TreeNode* p = tree;
    while (p != nullptr) {
        if (data > p->val) {
            if (p->right == nullptr) {
                p->right = new TreeNode(data);
                return;
            }
            p = p->right;
        } else { // data < p->val
            if (p->left == nulptr) {
                p->left = new TreeNode(data);
                return;
            }
            p = p->left;
        }
    }
}
```

**二叉查找树的删除操作**：

- 如果要删除的节点没有子节点，只需要直接将父节点中，指向要删除节点的指针置为 `nullptr`；
- 如果要删除的节点只有一个子节点（只有左子节点或者右子节点），只需要更新父节点中，指向要删除节点的指针，让它指向要删除节点的子节点；
- 如果要删除的节点有两个子节点，需要找到这个节点的**右子树中的最小节点**，把它的值替换到要删除的节点上，然后再利用上面两条规则来删除这个最小节点（最小节点肯定没有左子节点）；

```c++
void delete(int data) {
    TreeNode* p = tree;         // p 指向要删除的节点，初始化指向根节点
    TreeNode* pp = nullptr;     // pp 记录的是 p 的父节点
    while (p != nullptr && p->val != data) {
        pp = p;
        if (data > p->val) {p = p->right;}
        else {p = p->left;}
    }
    if (p == nullptr) {return;} // 	没有找到
    
    // 要删除的节点有两个子节点
    if (p->left != nullptr && p->right != nullptr) { // 查找右子树中最小节点
        TreeNode* minP = p->right;
        TreeNode* minPP = p;    // minPP 表示 minP 的父节点
        while (minP->left != nullptr) {
            minPP = minP;
            minP = minP->left;
        }
        p->val = minP->val;     // 将 minP 的数据替换到 p 中
        p = minP;               // 下面就是删除 minP
    }
    
    // 删除节点是叶子节点或者仅有一个节点
    TreeNode* chile;            // p 的子节点
    if (p->left != nullptr) {child = p->left;}
    eles if {p->right != nullptr} {child = p->right;}
    else {child = nullptr;}
    
    if (pp == nullptr) {tree = child;} // 删除的是根节点
    else if (pp->left == p) {pp->left = child;}
    else {pp->right = child;}
}
```

> 关于二叉树简单、取巧的方法：单纯将要删除的节点标记为「已删除」，并不真正从树中将这个节点删除。虽然比较浪费内存，但是在不增加插入、查找操作代码实现难度的条件下，使得删除操作变简单。

**二叉查找树的其他操作**：



## 链表（Linked-list）

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

## 常用函数

`__builtin_popcount()`：返回输入数据，二进制中「1」的个数，只对 `int` 类型；

如果想要对 `long int` 和 `long long` 类型使用，可以分别用 `__builtin_popcountl()` 和 `__builtin_popcountll`

`lower_bound(ForwardIterator first, ForwardIterator last, const T& val)`：返回一个迭代指针，该指针指向在 `[first, last)` 中不小于 `val` 的第一个元素。

`isalpha(c)`：判断 `c` 是否为一个字母。

**大小写转换**：`tolower(c)`：将字母转为小写字母；`toupper(c)`：将字母转为大写字母；

`to_string(c)`：将 `c` 转化为字符；

**排序**：

1. 对一个数组排序：`sort(nums.begin(), nums.end())`；

## 参考

[1] 《数据结构与算法之美》王争｜极客时间
