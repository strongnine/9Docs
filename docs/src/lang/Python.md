## 基础知识

### Python 列表的生成

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

## 迭代器

> 推荐阅读：
>
> - [Python: Built-in Types – Iterator Types](https://docs.python.org/3.8/library/stdtypes.html#iterator-types);
>
> - [Python：可迭代对象、迭代器、生成器函数、生成器的解析举例代码说明](https://blog.csdn.net/qq_41554005/article/details/119971444)；

迭代时 Python 范围集合元素的一种方法。

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

