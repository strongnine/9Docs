## 基础知识

关于 `__name__`：一个模块中，

- 如果直接运行文件 `__name__` 为 `__main__`；
- 如果该模块被调用，`__name__` 为被调用模块的模块名；

```python
# print_func.py 的代码如下
print('Hello World!')
print('__name__ value: ', __name__)
 
def main():
    print('This message is from main function')
 
if __name__ == '__main__':
   main()


# print_module.py 的代码如下
import print_func
print("Done!")


# 运行 print_module.py 的结果
>>> Hello World! __name__ value: print_func  Done! 
```

**py 文件执行完保持交互界面**：在终端用命令行 `python file.py` 执行 py 文件的时候，有时候想要继续测试代码，那么可以在文件的最后添加上下面的两行代码，这样在执行完 py 文件之后就会保持命令行交互界面不退出。

```python
# 执行完不退出 Python 交互
import code
code(banner="", local=locals())
```



为了提高内存利用效率，对于一些简单的对象，如一些数值较小的 int 对象（范围在 `[-5, 257)`），字符串对象等，Python 采用重用对象内容的方法。在 Python 3.6 中小整数对象池的范围会更大。

```python
a = [1, 2, 3]
b = [1, 2, 4]
id(a[1]) == id(b[1])  # 结果为 True

a[1] is b[1]  # 结果也为 True
# 1. is 比较两个对象的 id 值是否相等，是否指向同一个内存地址；
# 2. == 比较两个对象的内容是否相等，值是否相等
```

对于 `+=` 操作，如果是可变对象，则操作前后序列的 id 值不变，如果是不可变对象，则操作前后序列的 id 值会改变。

```python
# 列表是可变对象
lis = [1, 3, 2]
a = id(lis)
lis += [4, 5]
b = id(lis)
print(a == b)  # True

# 元祖是不可变对象
tup = (1, 3, 2)
a = id(lis)
tup += (4, 5)
b = id(tup)
print(a == b)  # False
```



```python
# 对于以下代码
# 1. 两个整数相除，结果为整数
# 2. 操作数之一是浮点，则两个数都转化为浮点计算
print type(1/2)  # Python 2.x
>>> <type 'int'>

# 无论是什么类型，都是按照正常的除法进行
print(type(1/2)) # Python 3.x
>>> <type 'float'>
```



`filter(object)`：将迭代器的数据代入函数中，返回使函数返回值为 True 的值

```python
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def is_even(n):
    return n % 2 == 0

print(list(filter(is_even(), a)))  # 会报错

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: is_even() missing 1 required positional argument: 'n'

# 正确的表达
print(list(filter(is_even, a)))  # 注意函数 is_even 后面不用加括号
# 不严谨地说, 类似于
[is_even(item) for item in a]
```



## 类

类变量：

- 类和实例都能访问；
- 通过类名修改类变量，会作用到所有的实例化对象；
- 通过类对象无法改变类变量。通过类对象对类变量赋值，本质不再是修改类变量的值，而是在给该对象定义新的实例变量。

```python
class Base(object):
    count = 0
    
    def __init__(self):
        pass
    
b1 = Base()
b2 = Base()
b1.count = b1.count + 1
print(b1.count, end=" ")
print(Base.count, end=" ")
print(b2.count)
>>> 1 0 0
```





## 函数

**Python 函数的闭包**：如果在函数中定义的 lambda 或者 def 嵌套在一个循环之中，而这个内嵌函数又引用了一个外层作用域的变量，该变量被循环所改变，那么所有在这个循环中产生的函数会有相同的值 —— 也就是在最后一次循环中完成时被引用变量的值。

```python
def fn():
    t = []
    i = 0  # 外层作用域
    while i < 2:
        t.append(lambda x: print(i * x, end=","))
        i += 1
    return t

for f in fn():  # fn() 执行完之后 i = 2
    f(2)  # t = [lambda x: print(2 * x, end=","), lambda x: print(2 * x, end=",")]
>>> 4,4,
```



若函数体内对一个变量重新赋值，会使得函数内部屏蔽了外面的全局变脸，导致报错

```python
>>> num = 1
>>> def fn():
...     num += 1
...     return lambda: print(num)
... 
>>> x = fn()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in fn
UnboundLocalError: local variable 'num' referenced before assignment
```





`*args` 和 `**kwargs` 是 Python 中方法的可变参数。`*args` 表示任何多个无名参数，是一个 Tuple；`**kwargs` 表示多个关键字参数，是一个 Dict。同时使用时 `*args` 参数要在 `**kwargs` 前面。当方法的参数不确定时，可以使用 `*args` 和 `**kwargs`.

```python
a, *b, c = range(5)  # *b: 剩下的参数会分配给 b
print(a, b, c)
>>> 0 [1, 2, 3] 4

*a, *b, c = range(5)  # 这种表达会报错
>>> SyntaxError: two starred expressions in assignment
```



Python 参数传递采用的是「传对象引用」的方式，这种方式相当于传值和传引用的一种综合。

- 如果函数收到的是一个可变对象（比如 `字典` 或者 `列表`）的引用，就能修改对象的原始值 —— 相当于通过「传引用」来传递对象。
- 如果函数收到的是一个不可变对象（比如 `数字`、`字符` 或者 `元组`）的引用，就不能直接修改原始对象 —— 相当于通过「传值」来传递对象。

```python
def changeList(nums):
    nums.append('c')
    print("nums", nums)

str1 = ['a', 'b']
# 调用函数
changeList(str1)
print("str1", str1)
>>> nums ['a', 'b', 'c'], str1 ['a', 'b', 'c']
```

Python 的默认参数只在函数定义时被赋值一次，而不会每次调用函数时又创建新的引用，函数定义完成后，默认参数已经存在固定的内存地址。

- 如果使用一个可变的默认参数并对其进行改变，那么以后对该函数的调用都会改变这个可变对象

- 默认参数如果是不可变对象，不存在该问题，每次调用都会将其变为默认值

```python
def fun(a = (), b = []):
    a += (1,)
    b.append(1)
    return a, b

fun()
print(fun())
>>> ((1,), [1, 1])
# !!! 注意 a == (1,)，而 b = [1, 1]
```



### 函数修饰器

**函数修饰符 `@`**：可以理解为引用、调用它修饰的函数

```python
def test(f):
    print("before ...")
    f()
    print("after ...")
    
@test
def func():
    print("func was called")

# 运行之后的输出结果为
>>> before ...
>>> func was called
>>> after ...
```

当 Python 解释器读到函数修饰符 `@` 的时候，执行的步骤为：

- 调用 test 函数，test 函数的入口参数就是 func 函数；
- test 函数被执行，入口参数的函数（func 函数）也会被调用；



```python
def dec(f):
    n = 3
    
    def wrapper(*args, **kw):
        return f(*args, **kw) * n
    
    return wrapper

@dec
def foo(n):
    return n * 2

foo(2) == 12  # True
foo(3) == 18  # True

# *args, **kw 是参数，用的是我们调用函数 foo(n) 时候的参数 n，
# 注意与 dec(f) 里的 n = 3 作区分
# 当我们利用了修饰符之后，就相当于
foo = dec(foo)
```



`@property`：相当于一个 get 方法，用于获取私有属性值，可以使得类有一个与方法同名的属性。

`@*.setter`：其中星号 `*` 代表方法名，它的两个作用是 1、对要存入的数据进行预处理；2、设置可读（不可修改）的属性。

> 注意：`@*.setter` 装饰器必须在 `@property` 装饰器的后面，并且两个被修饰的函数名词必须保持一致。



## 数据结构

### 列表

列表的切片一般指创造新的对象，是浅拷贝，不会有索引越界的情况，如果超出了列表的索引范围不会报错，会输出空列表。

```python
lists = [1, 2, 3, 4, 5, 6]
print(lists[6:])
>>> []
```



**Python 列表的生成**

**注意：在生成列表的时候，最好用 `[0 for _ in range(n)]` 的方式而不是 `[0] * n` 的方式生成，原因如下。**

假设我们想要创建一个长度为 `n = 2` 列表 `a` 的时候，一般的做法有两种：`a = [0] * n` 和 `a = [0 for _ in range(n)]`，在一维的时候两种方法没有区别。但是如果我们想要创建一个列表，列表中的每个元素都是长度为 `m = 3` 的列表时用两种方法出来的结果就是不同的：

```python
n, m = 2, 3
a = [[0] * n] * m  # 用第一种方法生成的
# a = [[0, 0], [0, 0], [0, 0]]
b = [[0 for _ in range(n)] for _ in range(m)]  # 用第二种方法生成的
# b = [[0, 0], [0, 0], [0, 0]]

a[0][0] = 1  # 令 a 列表的第一个元素为 1
# a = [[1, 0], [1, 0], [1, 0]]
b[0][0] = 1
# b = [[1, 0], [0, 0], [0, 0]]
```

我们的预期结果是像 `b` 这样的，也就是第一个列表的第一个元素等于 1，但是列表 `a` 是将每一个列表的第一个元素都设为 1 了。

导致这种情况的原因在于，用第一种生成方法是类似于 `=` 的方式去生成的，也就是假设 `a, b` 都是列表，我们令 `a = b `，如果他们其中一个的元素改变了，另一个也会跟着变，例如：

```python
a = [0, 0, 0]
b = a
b[0] = 1
# a = [1, 0, 0]
# b = [1, 0, 0]
```

要想避免这样的情况，就应该这样写 `b = list(a)`. 

回到一开始的问题，如果用第一种方法 `a = [[0] * n] * m`，那么 `a` 中的 `m` 个列表存储的地址都是一样的，那么你改变其中的一个列表，其他的列表都会跟着改变，就会出现上面的情况。

```python
c = [[0] * n for _ in range(m)]  # 没问题
d = [[0 for _ in range(n)]] * m  # 有问题
```

所以，以后要生成的列表时候，一律用 `[0 for _ in range(n)]` 会更好。



### 字符串

```python
str.strip()  # 删除首尾的空格
str.rstrip()  # 仅删除末尾的空格

str.upper()  # 把所有字符中的小写字母转化成大写字母
str.lower()  # 把所有字符中的大写字母转化成小写字母
str.capitalize()  # 把第一个字母转化为大写字母，其余小写
str.title()  # 把每个单词的第一个字母转化为大写，其余小写

str.find(char, beg=0, end=len(strs))  # 表示在 str 中返回第一次出现 char 的位置下标, 找不到返回 -1
# beg 表示在 strs 中的开始索引，默认为 0，end 为结束索引，默认为 strs 的长度。
str.index(char)  # 表示在 str 中返回第一次出现 char 的位置下标, 找不到会报错

str.rfind()  # 与 find 不同的在于它返回最后一次匹配的位置，如果匹配不到返回 -1

str.endswith(suffix[, start[, end]]) # 用于判断字符串是否以指定后缀结尾
# 如果以指定后缀结尾返回 True，否则返回 False
# start 与 end 为可选参数，代表检索字符串的开始和结束位置

str = "Hello, Python"
suffix = "Python"
print(str.endswith(suffix, 2))  # 从位置 2（'l'）开始判断字符串 str 是否以 suffix 结尾
>>> True
```



### 集合

集合（set）是一个无序的不重复元素序列

```python
# 集合 set 用大括号 {x, y,...} 或者 set() 来创建
# 注意：空集合的创建只能用 set(), {} 是创建空字典
s.add(x)  # 添加元素
s.update(x)  # 添加的元素可以是列表、元祖、字典等
s.remove(x)  # 移除元素, 如果元素不存在会报错
s.discard(x)  # 移除元素，不存在不会报错
s.pop()  # 随机删除集合中的一个元素
len(s)  # 计算集合元素的个数
s.clear()  # 清空集合
x in s  # 判断 x 是不是在集合中

# 如果集合 A 是集合 B 的子集，方法 issubset() 返回 True
# The issubset() method returns True if set A is the subset of B
A.issubset(B)
```



### 字典

字典是 Python里唯一的映射类型，它存储了键值对的关联，是由键到键值的映射关系。

```python
# 字典里面有一个 get 方法
dict.get(key, default)  # 当 key 对应的值存在时返回其本身，当 key 对应的值不存在时返回给定的 default 作为替代

dict.pop(key)  # 删除 key 对应的 value，并且返回该 value
```



#### 默认字典

defaultdict 中，当字典里的 key 不存在但被查找时，返回的不是 keyError 而是一个默认值。

```python
from collections import defaultdict  # 需要先导入
# 用法 dict_type = defaultdict( factory_function)
dict_int  = defaultdict(int)  # 不存在时返回 整数 0
dict_set  = defaultdict(set)  # 不存在时返回 空集 {}
dict_str  = defaultdict(str)  # 不存在时返回 空字符 ""
dict_list = defaultdict(list) # 不存在时返回 空列表 []
```





## 迭代器

> 推荐阅读：
>
> - [Python: Built-in Types – Iterator Types](https://docs.python.org/3.8/library/stdtypes.html#iterator-types);
>
> - [Python：可迭代对象、迭代器、生成器函数、生成器的解析举例代码说明](https://blog.csdn.net/qq_41554005/article/details/119971444)；

迭代是 Python 范围集合元素的一种方法。

**可迭代对象（Iterable）**：Python 中某对象实现 `__iter__()` 方法或者 `__getitem__()` 方法，且其参数从 0 开始索引，那么该对象就是可迭代对象。可以用 for 循环的对象，或者说**序列（Sequence）**都是可迭代对象，比如列表（list）、字典（dict）、元祖（tuple）、集合（set)、字符串（string）这些序列都是可迭代对象。

使用 `iter()` 方法可以将可迭代对象变成迭代器，如果可迭代对象实现了 `__iter__()` 方法，那么调用该方法会返回一个迭代器对象。调用迭代器的 `__next__()` 方法返回每一次迭代的内容，直到迭代完成后抛出 `StopIteration` 异常。

- 当使用 for 循环的时候，解释器会检查对象是否有 `__iter__()` 方法，有的话就调用它来获取一个迭代器；
- 如果没有 `__iter__()` 方法但是实现了 `__getitem__()`，解释器会创建一个迭代器，尝试从 0 开始按顺序遍历元素；
- 如果尝试失败，就会抛出一个 `TypeError`；
- 字符串、列表、元祖、字典、集合等均不是迭代器，但是他们是可迭代对象，在迭代中本质上就是对调用 `__iter__()` 后得到的迭代器通过不断使用 `next()` 函数来实现的；
- 使用  `isinstance()` 函数可以判断某个对象是否为某一类型，因此可以使用 `isinstance(nums, Iterable)` 来判断对象 `nums` 是否为可迭代对象；

**迭代器（Iterator）**：可以记住遍历对象的位置，其内部有一个状态用于记录迭代所在的位置，以便下次迭代时能够取出正确的元素。

- 迭代器有两个基本方法 `iter()` 和 `next()`，前者可以将可迭代对象变成迭代器，后者可以返回下一个值；
- 要定义一个迭代器必须实现 `__iter__()`  和 `__next__()` 方法。在 Python 2 中则要求类内包含有 `next()` 方法；
- 迭代器只能往前不能后退，从集合的第一个元素开始访问，直到所有的元素被访问完之后结束；
- 迭代器一定是可迭代对象，但是可迭代对象不一定是迭代器，例如字符串、字典、元祖、集合等；

```python
>>> nums = [1, 2, 3]   # 创建一个列表
>>> nums_iterator = iter(nums)  # 得到一个迭代器
>>> print(nums_iterator)
<list_iterator object at 0x7fe1a014e250>
>>> print(next(nums_iterator))
1
>>> print(next(nums_iterator))
2
>>> print(next(nums_iterator))
3
# next() 方法调用到末尾时会跳出 StopIteration
>>> print(next(nums_iterator))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

创建一个迭代器的具体方式如下所示，该代码实现的迭代器功能为迭代从 0 每次递增 1 到 9 的所有数字：

```python
class my_inter:
    def __iter__(self):
        self.length = 10 
        self.index = 0
        return self


    def __next__(self):
        if self.index < self.length:
            x = self.index
            self.index += 1
            return x
        else:
            raise StopIteration


nums = my_inter()  # 实例化对象
print(nums)
# 对实例化对象使用 iter() 返回迭代器
nums_iter = iter(nums)

for x in nums_iter:  # 对迭代器进行迭代
    print(x, end=" ")

>>> 0 1 2 3 4 5 6 7 8 9 
```

## 生成器

在 Python 中使用了 `yield` 的函数称为**生成器函数（Generator Function）**。调用一个生成器函数，返回的是一个实例化迭代器对象。生成器函数返回一个只能用于迭代操作的**生成器（Generator）**，生成器是一种特殊的迭代器，自动实现了「迭代器协议」，即实现 `__iter__()` 和 `__next__()` 两个方法，无需再手动实现。

生成器因为有 `send()` 方法，因此在迭代的过程中可以改变当前迭代值，而这在普通迭代器上会引发异常。在调用生成器运行的过程中，每次遇到 `yield` 时函数会暂停并保存当前所有的运行信息，返回 `yield` 语句表达式的值，并在下一次执行 `next()` 或者 `send()` 方法时从当前位置继续运行。

> 注意：在 Python 3 开始，生成器的 `next()` 方法变成 `__next__()`

使用以下方法判断是否为生成器函数或者是否为生成器：

```python
from inspect import isgeneratorfunction
isgeneratorfunction(x)  # 判断 x 是否为生成器函数

import types
isinstance(x, types.GeneratorType)  # 判断 x 是否为生成器
```

创建一个生成器函数的具体方式如下所示：

```python
def my_list(num):   # 定义生成器
    now = 0   # 当前迭代值，初始为 0
    while now < num:
        val = yield now  # 返回当前迭代值，并接受可能的 send 发送值
        # val 如果为空，迭代值自增 1；否则重新设定当前迭代值为 val
        now = now + 1 if val is None else val


addOneGenera = my_list(5)  # 得到一个生成器对象
print("下一个迭代值：{}".format(addOneGenera.__next__()))
print("下一个迭代值：{}".format(addOneGenera.__next__()))
print("重新设定当前迭代值为 3")
addOneGenera.send(3)
print("下一个迭代值：{}".format(addOneGenera.__next__()))

>>> 下一个迭代值：0
>>> 下一个迭代值：1
>>> 重新设定当前迭代值为 3
>>> 下一个迭代值：4
```

## 虚拟环境 virtualenv

pip, virtualenv, fabric 统称为 Python 的三大神器。其中 `virtualenv` 的作用是建立一个虚拟的 Python 环境。

通过 pip 安装 virtualenv：`pip install virtualenv`，如果输入 `virtualenv --version` 能够输出版本号就代表安装成功了。

**为项目搭建新的虚拟环境**：`virtualenv nine-py`，执行完之后会在当前的目录中创建一个相对应名字的文件夹，是独立的 Python 运行环境，包含了 Python 可执行文件，以及 pip 库的一份拷贝，在这个环境中安装的库都是独立的，不会影响到其他的环境。

> 如果想要指定 Python 解释器：`virtualenv -p /usr/bin/python2.7 nine-py`

**激活虚拟环境**：`source nine-py/bin/activate`

**停用虚拟环境**：`deactivate`（停用之后会回到系统默认的 Python 解释器）

**查看当前安装版本**：`pip freeze`

**将当前环境输出为文件**：`pip freeze > requirements.txt`，会创建一个 requirements.txt 文件，其中包含当前环境所有包以及对应版本的简单列表。

**安装环境文件**：`pip install -r requirements.txt`

## 文件操作

### 读取文件

```python
# 读取文件的不同方法
read(size)  # 从文件当前位置起读取 size 个字节，若不给定参数则读取至文件末尾
readline()  # 每次读出一行内容，占用内存小，适合读取大文件
readlines()  # 读取文件所有行，保存在一个 list 中
```



## Socket 操作

`sk.recv(bufsize[,flag])`：接受套接字的数据。数据以字符串形式返回，bufsize 指定最多可以接收的数量。flag 提供有关消息的其他信息，通常可以忽略。

`sk.recvfrom(bufsize[.flag])`：与 `recv()` 类似，但返回值是 ` (data, address)`。其中 data 是包含接收数据的字符串，address 是发送数据的套接字地址。

`sk.getsockname()`：返回套接字自己的地址。通常是一个元组 `(ipaddr, port)`。

`sk.connect(address)`：连接到 address 处的套接字。一般，address 的格式为元组 `(hostname, port)`，如果连接出错，返回 socket.error 错误。

`sk.listen(backlog)`：开始监听传入连接。backlog 指定在拒绝连接之前，可以挂起的最大连接数量。

