# 数据结构

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

## 向量（vector）

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

## 栈（stack）





## 队列（queue）

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





## 链表（linked-list）

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

`tolower(c)`：将字母转为小写字母；`toupper(c)`：将字母转为大写字母；

`to_string(c)`：将 `c` 转化为字符；
