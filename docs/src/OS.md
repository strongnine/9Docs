## 操作系统

> 资料：
>
> - [B 站课程：南京大学 2022 操作系统 - 蒋炎岩](https://www.bilibili.com/video/BV1Cm4y1d7Ur/?spm_id_from=333.999.0.0&vd_source=303712e6e1938f6c6101fba5372c11f9)；课程链接：[操作系统：设计与实现 (2022 春季学期)](http://jyywiki.cn/OS/2022/)；
> - MIT - The Missing Semester of Your CS Education；

**操作系统（Operating System, OS）**：是管理计算机硬件和软件资源的计算机程序，提供一个计算机用户与计算机硬件系统之间的接口。向上对用户程序提供接口，向下接管硬件资源。操作系统本质上也是一个软件，作为最接近硬件的系统软件，负责处理器管理、存储器管理、设备管理、文件管理和提供用户接口。

- 操作系统服务谁？程序 = 状态机；涉及多线程 Linux 应用程序；
- 操作系统为程序提供什么服务？操作系统 = 对象 + API（应用视角/设计）；涉及 POSIX + 部分 LInux 特性；
- 如何实现操作系统提供的服务？操作系统 = C 程序（硬件视角/实现）；涉及 xv6、自制迷你操作系统；

计算机专业学生必须具备的核心素质：

- 是一个合格的操作系统用户：
  - 会 STFW/RTFM 自己动手解决问题
  - 不怕使用任何命令行工具 `vim`, `tmux`, `grep`, `gcc`, `binuils`, ….
- 不惧怕写代码：
  - 能管理一定规模（数千行）的代码；
  - 能在出 Bug 时默念「机器永远是对的、我肯定能调出来的」，然后开始用正确的工具/方法调试；

**❓什么是源代码视角下的程序？**

数字逻辑电路：

- 状态 = 寄存器保存的值（flip-flop）；
- 初始状态 = RESET（implementation dependent）；
- 迁移 = 组合逻辑电路计算寄存器下一周期的值；

如果把状态机用一段代码写出来，是这样的

```c
#include <stdio.h>
#include <unist.h> 

#define REGS_FOREACH(_) _(X) _(Y)
#define RUN_LOGIC  X1 = !X && Y; \
				   Y1 = !X && !Y;
#define DEFINE(X)  static int X, X##1;
#define UPDATE(X)  X = X##1;
#define PRINT(X)   printf(#X " = %d; ", X);

int main() {
    REGS_FOREACH(DEFINE);
    while (1) {  //clock
    	RUN_LOGIC;
        REG_FOREACH(PRINT);
        REG_FOREACH(UPDATE);
        putchar('\n'); sleep(1);
    }
}
```

更完整的一种实现是「数码管显示」，输出数码管的配置信号：

```c
#include <stdio.h>
#include <unistd.h>

#define REGS_FOREACH(_)  _(X) _(Y)
#define OUTS_FOREACH(_)  _(A) _(B) _(C) _(D) _(E) _(F) _(G)
#define RUN_LOGIC        X1 = !X && Y; \
                         Y1 = !X && !Y; \
                         A  = (!X && !Y) || (X && !Y); \
                         B  = 1; \
                         C  = (!X && !Y) || (!X && Y); \
                         D  = (!X && !Y) || (X && !Y); \
                         E  = (!X && !Y) || (X && !Y); \
                         F  = (!X && !Y); \
                         G  = (X && !Y); 

#define DEFINE(X)   static int X, X##1;
#define UPDATE(X)   X = X##1;
#define PRINT(X)    printf(#X " = %d; ", X);

int main() {
  REGS_FOREACH(DEFINE);
  OUTS_FOREACH(DEFINE);
  while (1) { // clock
    RUN_LOGIC;
    OUTS_FOREACH(PRINT);
    REGS_FOREACH(UPDATE);
    putchar('\n');
    fflush(stdout);
    sleep(1);
  }
}
```

> 在[课程课件](http://jyywiki.cn/OS/2022/slides/2.slides#/1/3)中复制 `logisim.c` 代码的下载链接使用 `wget http://jyywiki.cn/pages/OS/2022/demos/logisim.c` 将文件下载，然后使用 `vi logisim.c` 可以查看和编辑。在 vim 下用 `:!gcc %` 可以编译当前文件，没有报错可以 `:!./a.out` 来看看程序的运行输出结果。
>
> 在课程课件中复制下载 `seven-seg.py` 代码：`wget http://jyywiki.cn/pages/OS/2022/demos/seven-seg.py`，然后再次运行 `./a.out | python seven-seg.py`，就会出现数码管的显示了；
>
> `seven-seg.py` 将 7 个晶体管用 A - G 分别表示，然后用 1 表示亮，0 表示不亮；

C 程序的状态机：

- 状态  = 堆 + 栈
- 初始状态 = `main` 的第一条语句；
- 迁移 = 执行一条简单语句
  - 任何 C 程序都可以改写成「非复合语句」的 C 代码；
  - 真的有这种工具（C Intermediate Language）和解释器；

‼️任何真正的理解都应该落到可以执行的代码。

下载课件 [`hanoi-r.c`](http://jyywiki.cn/pages/OS/2022/demos/hanoi-r.c) 为其写一个 `main` 函数：

```c
#include <stdio.h>
#include "hanoi-r.c"

int main() {
    hanoi(3, 'A', 'B', 'C');
    return 0;
}
```

编译并执行 `gcc main.c && ./a.out`，可以看到函数的输出。使用 gdb 运行程序：`gdb ./a.out` 可以看到堆栈情况和单步执行。命令：`start` 程序开始，`step` 单步执行，`info frame` 打印堆栈信息。

> M1 Mac ARM 架构目前无法安装 gdb。

**❓什么是 lldb 和 gdb？它们有什么区别？**

C 程序的状态机模型：

- 状态 = stack frame 的列表（每个 frame 有 PC）+ 全局变量；
- 初始状态 = `main(argc, argv)`，全局变量初始化；
- 迁移 = 执行 `top stack frame PC` 的语句；PC++。PC 是指向程序语句的指针；
  - 函数调用 = push frame（frame.PC = 入口）；
  - 函数返回 = pop frame；

**❓什么是二进制视角下的程序？**

还是状态机：

- 状态 = 内存 M + 寄存器 R；（寄存器在 CPU 上，本质也是一个内存）
- 初始状态 = 
- 迁移 = 执行一条指令

调用操作系统 `syscall`；

- 把 $(M,R)$ 完全交给操作系统，任其修改
  - ❓如果程序不打算完全信任操作系统呢？
- 实现与操作系统中的其他对象交互
  - 读写文件/操作系统状态（例如把文件内容写入 $M$）
  - 改变进程（运行中状态机）的状态，例如创建进程/销毁自己

**❓既然 <程序 = 计算 + syscall>，请问如何构造一个最小的 Hello, World？**

用 `gcc` 编译出来的文件不满足「最小」

- `--verbose` 可以查看所有编译选项（有很多）。`printf` 变成了 `puts@plt`；
- `-static` 会复制 `libc`；
- 强行编译 + 链接：`gcc -c` + `ld`；链接失败 `ld` 不知道怎么链接库函数；如果把 `printf` 函数去掉，即定义空的 `main` 函数（把 `main` 改成 `_start` 可以避免一个奇怪的警告），这个时候会出现 Segmentation Fault；

下载最小版本的汇编代码 [`minimal.S`](http://jyywiki.cn/pages/OS/2022/demos/minimal.S)

```assembly
#include <sys/syscall.h>

.globl _start
_start:
  movq $SYS_write, %rax   # write(
  movq $1,         %rdi   #   fd=1,
  movq $st,        %rsi   #   buf=st,
  movq $(ed - st), %rdx   #   count=ed-st
  syscall                 # );

  movq $SYS_exit,  %rax   # exit(
  movq $1,         %rdi   #   status=1
  syscall                 # );

st:
  .ascii "\033[01;31mHello, OS World\033[0m\n"
ed:
```

gcc 支持对汇编代码的预编译（还会定义 `__ASSEMBLER__` 宏）

编译命令：`gcc minimal.S -c && ld minimal.o`

**❓为什么会出现 Segmentation Fault？**

可以观察程序（状态机）的执行：

- 初学者必须克服的恐惧：`STFW/RTFM`（M 非常有用）；
- `starti` 可以帮助从第一条指令开始执行程序；【可以去看看 GDB 的官方文档】
- `gdb` 可以在两种状态机视角之间切换（layout）；

因为返回的是初始状态，状态机的初始状态是不能够返回的；

**❓杀人面试题 (1)：一个普通的、人畜无害的 Hello World C 程序执行的第一条指令在哪里？**

可以用 `gdb a.out`，然后 `starti` 去调试。`info proc {mappings,...}` 打印进程内存。

[`main()` 之前发生了什么？](http://jyywiki.cn/OS/2022/slides/2.slides#/5/5)

**❓[杀人的面试题 (2)：main 执行之前、执行中、执行后，发生了哪些操作系统 API 调用？](http://jyywiki.cn/OS/2022/slides/2.slides#/5/6)**

**❓有什么办法让状态机「停下来」？**

如果是纯「计算」 的状态机，是不行的，要么是死循环，要么 undefined behavior；

解决办法是使用 `syscall`

```c
#include <sys/syscall.h>

int main() {
    syscall(SYS_exit, 42);
}
```

一些彩蛋：ANSI Escape Code. 在终端中一些字符可以有颜色，特殊编码的字符实现终端控制

- `vi.c from busybox`；
- `telnet towel.blinkenlights.nl`（可以看到用字符做成的电影，按 `Ctrl + ] and q` 退出）；
- `dialog --msgbox 'Hello, OS World!' 8 32`；
- `ssh sshtron.zachlatta.com`（网络游戏）；

**❓什么是编译器？什么是正确的编译？**

正确的编译就一句话：在 C 代码中所有不可优化的代码都被翻译到汇编语言中；

同步（Synchronization）与异步（Asynchronous）。

生产者 - 消费者问题：99% 的实际并发问题都可以用生产者 - 消费者解决。

```c
void Tproduce() { while (1) printf("("); }
void Tconsume() { while (1) printf(")"); }
```

这个代码需要解决的问题是：

- 保证打印出来的括号是合法的，或者是某个合法括号序列的前缀；
- 括号嵌套的深度不超过 $n$；

实现同步的方法：（1）等到有空位的时候再打印左括号；（2）等到能够配对的时候再打印右括号。因此：

- 左括号：相当于生产资源（任务）、放入队列；
- 右括号：从队列取出资源（任务）执行；

使用互斥锁实现括号问题：

- 左括号：嵌套深度（队列）不足 $n$ 的时候才能够打印：
- 右括号：嵌套深度（队列）$> 1$ 时候才能够打印；

条件变量（Conditional Variables, CV）：

- `wait(cv, mutex)`：调用时必须保证已经获得 `mutex`；释放 `mutex` 进入睡眠状态。在睡眠状态的时候，系统不会对代码有任何操作；
- `signal/notify(cv)`：如果有线程正在等待 `cv`，则唤醒其中一个线程；
- `broadcast/notifyAll(cv)`：唤醒全部正在等待 `cv` 的线程；

❓线程（Threads）和协程（Corotines）

❓互斥锁和自旋锁的区别

## 堆栈

堆和栈的区别：

- 栈区（stack）：由编译器自动分配和释放，存放函数的参数值、局部变量的值等。其操作方式类似于数据结构的栈；
- 堆区（heap）：一般由程序员分配和释放，若程序员不释放，程序结束时可能由操作系统回收。需要手动使用 `new`, `malloc`, `delete`, `free` 进行分配和回收，空间较大，但可能会出现内存泄漏和空闲碎片的情况。它与数据结构中的堆是两回事，分配方式类似于链表；

- 申请方式：

  - 栈：**由系统自动分配。**例如在声明函数的一个局部变量 `int b`，系统会自动在栈中为 `b` 开辟空间；

  - 堆：**需要程序员自己申请，并指明大小。**在 C 中用 `maclloc` 函数申请，在 C++ 中用 `new` 操作符；

- 申请后系统的响应：
  - 栈：只要栈的剩余空间大于所申请的空间，系统将为程序提供内存，否则将报异常提示栈溢出；
  - 堆：操作系统有一个记录空间内存地址的链表，当系统收到程序的申请时，会遍历链表，寻找第一个空间大于所申请空间的堆结点，然后将节点从内存空闲节点链表中删除，并将该节点的空间分配给程序。对于大多数操作系统，会在这块内存空间中的首地址处记录本次分配的大小。这样，代码中的 `delete` 语句才能正确地释放本地内存空间。另外，由于找到的堆节点的大小不一定正好等于申请的大小，系统会自动地将多余的那部分重新放入到链表中。
- 申请大小的限制：
  - 栈：在 Windows 下，栈是向低地址拓展的数据结构，是一块连续的内存的区域。栈的地址和栈的大小是系统预先规定好的，如果申请的内存空间超过栈的剩余空间，将提升栈溢出。**栈的空间有限**；
  - 堆：是向高地址拓展的内存结构，是不连续的内存区域。是系统用链表存储空闲内存地址的，不连续。**堆是很大的自由内存区**；
- 申请效率的比较：
  - 栈：由系统自动分配，速度较快，但程序员无法控制；
  - 堆：由 `new` 分配的内存，一般速度比较慢，而且容易产生内存碎片，不过用起来方便；
  - 拓展：在 Windows 操作系统中，最好的方式是用 `VirtualAlloc` 分配内存。不是在堆，不是在栈，而是在内存空间中保留一块内存，虽然用起来不方便，但是速度快，也很灵活；
- 堆和栈的存储内容：
  - 栈：在函数调用时，第一个进栈的是主函数中的下一条指令（函数调用的下一个可执行语句）的地址，然后是函数的各个参数。在 C 编译器中，参数是由右往左入栈的，然后是函数的局部变量，静态变量不入栈；
  - 堆：一般是在堆的头部用一个字节存放堆的大小。堆中的具体内容由程序员安排。

- 总结：
  - 栈的空间由操作系统自动分配以及释放；堆的空间是程序员手动分配以及释放的；
  - 栈的空间有限；堆是很大的自由内存区；
  - C 中的 `maclloc` 函数分配的内存空间就在堆上；C++ 中对应的是 `new` 操作符；

除了栈区和堆区之外，还有：

- 全局/静态存储区：全局变量和静态变量的存储是放在一起的，初始化的全局变量和静态变量在一块区域，未初始化的全局变量和未初始化的静态变量在相邻的另外一块区域。程序结束后由系统释放；
- 文字常量区：常量字符串存放的位置，一般不允许修改，程序结束之后由系统释放；
- 程序代码区：存放函数体的二进制代码；

❓什么是内存泄漏？要怎么排查？

## 命令行

`uname`：查询系统基本信息的命令：

- `-s, --kernel-name` 输出内核名称；
- `-n, --nodename` 输出网络节点上的主机名；
- `-r, --kernel-release` 输出内核发行号；
- `-v, --kernel-version` 输出内核版本；
- `-m, --machine` 输出主机的硬件架构名称；
- `-p, --processor` 输出处理器类型或 `unknown`；
- `-i, --hardware-platform` 输出硬件平台或 `unknown`；
- `-o, --operating-system` 输出操作系统名称；

❓`uname -a` 输出的是什么？

```bash
(base) strongnine@strongs-MacBook-Pro opt % uname -a
Darwin strongs-MacBook-Pro.local 22.1.0 Darwin Kernel Version 22.1.0: Sun Oct  9 20:14:30 PDT 2022; root:xnu-8792.41.9~2/RELEASE_ARM64_T8103 arm64
```

❓`tldr` (Too Long; Didn’t Read) 是做什么的？

`wget`：

**❓什么是 `strace`？**

`strace` 也即是 system call trace。在理解程序运行时使用的系统调用，例如 `strace ./hello-goodbye`。

所有的程序都是状态机，它们不断地在「计算」和「syscall」两种状态转换；

- 被操作系统加载：通过另一个进程执行 `execve` 设置为初始状态；
- 状态机执行：
  - 进程管理：`fork`, `execve`, `exit`, …
  - 文件/设备管理：`open`, `close`, `read`, `write`, …
  - 存储管理：`mmap`, `brk`, …
- 直到 `_exit (exit_group)` 退出；

在不同的视角下去理解程序

- 源代码 S：状态迁移 = 执行语句；
- 二进制代码 C：状态迁移 = 执行指令；
- 编译器 C = compile(S)；
- 应用视角下的 OS：就是一条 syscall 指令；
- 理解 OS 的重要工具：`gcc`, `binutils`, `gdb`, `strace`； 

## 并发编程

并发（concurrency）：如果逻辑控制流在时间上重叠，那么它们就是并发的（concurrent），这种常见的现象称为并发。

并发程序（concurrent program）：使用应用级并发的应用程序称为并发程序。现代操作系统提供了山中基本的构造并发程序的方法：

- 进程：每个逻辑控制流都是一个进程，由内核来调度和维护。进程有独立的虚拟地址空间，要想和其他流通信，控制流必须使用某种显式的进程间通信（interprocess communication, IPC）机制。
- I/O 多路复用：应用程序在一个进程的上下文中显式地调度它们自己的逻辑流。逻辑流被模型化为状态机，数据到达文件描述符后，主程序显式地从一个状态转换到另一个状态。因为程序是一个单独的进程，所以所有的流都共享同一个地址空间。
- 线程：运行在一个单一进程上下文中的逻辑流，由内核进行调度，可以看成是其他两种方式的混合体，像进程流一样由内核进行调度，又像 I/O 多路复用流一样共享同一个虚拟地址空间。

### 进程

进程的优点：

- 有独立的地址空间，进程不会不小心覆盖另一个进程的虚拟内容。

进程的缺点：

- 独立的地址空间使得进程共享状态信息更加困难，需要使用显式的 IPC（进程间通信）机制，而这开销很大。

> Unix IPC 通常指的是所有允许进程和同一台主机上其他进程进行通信的技术，包括：管道、先进先出（FIFO）、系统 V 共享内存、系统 V 信号量（semaphore）。

### I/O 多路复用

I/O 多路复用的优点：

- 比基于进程的涉及给了程序员更多的对程序行为的控制。
- 基于 I/O 多路复用的事件驱动器是运行在单一进程上下文中的，因此每个逻辑流都能访问该进程的全部地址空间。
- 因为不需要进程上下文切换来调度新的流，事件驱动设计常常比基于进程的设计要高效得多。

I/O 多路复用的缺点：

- 编码复杂，并且随着并发粒度的减小，复杂性变高。
- 不能充分利用多核处理器。

> 现代高性能服务器（如 Node.js, nginx, Tornado）使用的都是基于 I/O 多路复用的事件驱动的编程方式，因为相比较于进程和线程的方式，它有明显的性能优势。
>
> 并发粒度：是指每个逻辑流每个时间片执行的指令数量。

### 线程

线程（thread）：运行在进程上下文中的逻辑流。每个线程都有自己的线程上下文（thread context），包括一个唯一的整数线程 ID（Thread ID, TID）、栈、栈指针、程序计数器、通用目的寄存器和条件码。所有的运行在一个进程里的线程共享该进程的整个虚拟地址空间。

主线程（main thread）：每个进程开始生命周期时都是单一线程，这个线程称为主线程。在某一时刻，主线程创建一个对等线程（peer thread），从这个时间点开始，两个线程就并发地运行。

因为一个线程的上下文要比一个进程的上下文小得多，线程的上下文切换要比进程的上下文切换快得多。

在任何一个时间点上，线程是可结合的（joinable）或者是可分离的（detached）。一个

### 并发与并行

操作系统是最早的并发程序之一。并发的基本单位是「线程」。

原子性：一段代码执行（例如 `pay()`）独占整个计算机系统。「程序（甚至是一条指令）独占处理器执行」的基本假设在现代多处理器系统上不再成立。

99% 的并发问题都可以用一个队列解决：

- 把大任务切分成可以并行的小任务；
- worker thread 去锁保护的队列里取任务；
- 除去不可并行的部分，剩下的部分可以获得线性的加速度：$T_n<T_{\infty}+\frac{T_1}{n}$

单个处理器把汇编代码（用电路）编译成更小的 $\mu$ops，每个 $\mu$ops 都有 Fetch、Issue、Execute、Commit 四个阶段。在任何时刻，处理器都维护一个 $\mu$ops 的「池子」

- 每一周期向池子补充尽可能多的 $\mu$op：多发射；
- 每一周期（在不违反编译正确性的前提下）执行尽可能多的 $\mu$op：乱序执行、按序提交；

**❓并发和并行有什么区别？**

- 并发（concurrency），指的是多个事情，在**同一时间段**内同时发生了。例如在单核 CPU 上的多任务，在宏观上看来两个程序是同时运行的，但是从微观上看两个程序的指令是交织着运行的，在单个周期内只运行了一个指令；
- 并行（parallelism），指的是多个事情，在**同一时间点**上同时发生了。是严格物理意义上的同时运行，比如两个程序分别运行在两个 CPU 核心上，两者之间相互不影响，单个周期内每个人程序都运行了自己的指令；
- 只有在多 CPU 的情况下，才会发生并行，否则看似同时发生的事情，其实都是并发执行的；
- 并发并不会提高计算机的性能，只能够提高效率，例如降低某个进程的相应时间。而并行可以提高计算机的性能。

互斥：保证两个线程不能同时执行一段代码

## 问题

❓ARM 和 X86 架构有什么区别？

中央处理单元（CPU）主要由运算器、控制器、寄存器三部分组成，从字面意思看运算器就是起着运算的作用，控制器就是负责发出 CPU 每条指令所需要的信息，寄存器就是保存运算或者指令的一些临时文件，这样可以保证更高的速度。

CPU 有着处理指令、执行操作、控制时间、处理数据四大作用。

CPU 的架构可以分为两大类：复杂指令集和精简指令集，即 CISC 和 RISC。Intel 的处理器使用的是复杂指令集（CISC），ARM 处理器使用精简指令集（RISC）。ARM 架构过去称作进阶精简指令集机器（AdvancedRISCMachine），更早称作：AcornRISCMachine。

ARM 从来只是设计低功耗处理器，Intel 的强项是设计超高性能的台式机和服务器处理器。

Intel 并没有开发 64 位版本的 x86 指令集。64 位的指令集名为 x86-64（有时简称为 x64），实际上是 AMD 设计开发的。Intel 想做 64 位计算，它知道如果从自己的 32 位 x86 架构进化出 64 位架构，新架构效率会很低，于是它搞了一个新 64 位处理器项目名为 IA64。由此制造出了 Itanium 系列处理器。同时 AMD 知道自己造不出能与 IA64 兼容的处理器，于是它把 x86 扩展一下，加入了 64 位寻址和 64 位寄存器。最终出来的架构，就是 AMD64，成为了 64 位版本的 x86 处理器的标准。

ARM 的 big.LITTLE 架构是一项 Intel 一时无法复制的创新。在 big.LITTLE 架构里，处理器可以是不同类型的。传统的双核或者四核处理器中包含同样的 2 个核或者 4 个核。一个双核 Atom 处理器中有两个一模一样的核，提供一样的性能，拥有相同的功耗。ARM 通过 big.LITTLE 向移动设备推出了异构计算。这意味着处理器中的核可以有不同的性能和功耗。当设备正常运行时，使用低功耗核，而当你运行一款复杂的游戏时，使用的是高性能的核。

设计处理器的时候，要考虑大量的技术设计的采用与否，这些技术设计决定了处理器的性能以及功耗。在一条指令被解码并准备执行时，Intel和ARM的处理器都使用流水线，就是说解码的过程是并行的。

为了更快地执行指令，这些流水线可以被设计成允许指令们不按照程序的顺序被执行（乱序执行）。一些巧妙的逻辑结构可以判断下一条指令是否依赖于当前的指令执行的结果。Intel和ARM都提供乱序执行逻辑结构，可想而知，这种结构十分的复杂，复杂意味着更多的功耗。

Intel 处理器由设计者们选择是否加入乱序逻辑结构。异构计算则没有这方便的问题。ARM Cortex-A53 采用顺序执行，因此功耗低一些。而 ARM Cortex-A57 使用乱序执行，所以更快但更耗电。采用 big.LITTLE 架构的处理器可以同时拥有 Cortex-A53 和 Cortex-A57 核，根据具体的需要决定如何使用这些核。在后台同步邮件的时候，不需要高速的乱序执行，仅在玩复杂游戏的时候需要。在合适的时间使用合适的核。

❓从精度、非空、非负考虑，采用 `float`

面试官提出 `float` 精度问题，引申到存储原理，如何判 0？

❓`float` 数 (1 - 0.9) 与 (0.9 - 0.8) 相等吗？面对精度丢失，如何改进？引申到整型，把余额×100，转整型；

❓如何给文件增加运行权限？

❓Linux 软连接和硬连接的区别？

❓线程的共享资源和私有资源是什么？

❓什么是内存池？怎么去设计？怎么去测试性能会提升多少？

❓内存池跑在 32 位系统和 64 位系统上可能会有什么问题？

❓内存池会不会无限扩展？

❓回收内存的时候，如何确定回收的块的大小？

❓select、poll、epoll 有什么区别？

## 资料

https://gitee.com/autoencoder/interviewtop/blob/master/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F.md

[操作系统八股文背诵版 - 互联网面试八股文](https://zhuanlan.zhihu.com/p/373966882)
