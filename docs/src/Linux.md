## Linux 系统

Linux 系统可以划分为 4 部分：Linux 内核、GNU 工具、图形化桌面环境、应用软件。

Linux 内核主要负责 4 种功能：

- 系统内存管理
- 软件程序管理
- 硬件设备管理
- 文件系统管理

**常见的 Linux 目录名**：

| 目录     | 用途                                                         |
| -------- | ------------------------------------------------------------ |
| /        | 虚拟目录的根目录，通常不会在这里放置文件                     |
| /bin     | 二进制文件目录，存放了很多用户级的 GNU 实用工具              |
| /boot    | 引导目录，存放引导文件                                       |
| /dev     | 设备目录，Linux 在其中创建设备节点                           |
| /etc     | 系统配置文件目录                                             |
| /home    | 主目录，Linux 在其中创建用户目录（可选）                     |
| /lib     | 库目录，存放系统和应用程序的库文件                           |
| /libname | 库目录，存放替代格式的系统和引用程序库文件（可选）           |
| /media   | 媒介目录，可移动存储设备的常用挂载点                         |
| /mnt     | 挂载目录，可用于临时挂载文件系统的常用挂载点                 |
| /opt     | 可选目录，存放第三方软件包                                   |
| /proc    | 进程目录，存放现有内核，系统以及进程的相关信息               |
| /root    | root 用户的主目录（可选）                                    |
| /run     | 运行目录，存放系统的运行时数据                               |
| /sbin    | 系统二进制文件目录，存放了很多管理级的 GNU 实用工具          |
| /srv     | 服务目录，存放本地服务的相关文件                             |
| /sys     | 系统目录，存放设备、驱动程序以及部分内核特性信息             |
| /tmp     | 临时目录，可以在其中创建和删除临时工作文件                   |
| /usr     | 用户目录，一个次目录层级结构（secondary directory hierarchy） |

文件系统层级标准（filesystem hierarchy standard, FHS）：常见的 Linux 目录名均基于文件系统层级标准。

`cd [路径]`：切换路径。单点号 `.` 表示当前目录，双点号 `..` 表示当前目录的父目录。

`pwd`：显示 shell 会话的当前目录。

列表命令 `ls`：

- `ls -F` 可以区分文件和目录。目录名之后会带有 `/`，在可执行文件之后带有星号 `*`；
- `ls -a` 可以将以单点号 `.` 开始的隐藏文件也显示出来；
- `ls -R` 递归显示，可以列出当前目录所包含的子目录中的文件。如果子目录很多输出的内容会变得很长；
- `ls -l` 显示更多的目录和文件信息。例如：`drwxr-xr-x. 2 strongnine 		strongnine 6 Feb 20 14:23 Desktop`，其中包括的信息有：
  - 文件类型：目录（d）、文件（-）、链接文件（l）、字符设备（c）、块设备（b）
  - 文件的权限、文件的硬链接数、文件属主、文件属组、文件大小（以字节为单位）
  - 文件的上次修改时间、文件名或目录名

> - 选项参数可以结合使用，例如 `ls -alF`。
> - 如果想查看单个文件的长列表，只需要在 `ls -l` 命令之后跟上该文件名即可。

过滤器：就是一个字符串，可以用作简单的文本匹配。

-  问号 `?` 可以代表过滤器字符串中任意位置的单个字符；
- 星号 `*` 可以用来匹配零个或多个字符；
- 方括号 `[a-z]` 可以指定要匹配的字母或者范围；
- 惊叹号 `!` 可以将不希望出现的内容排除在外；

通配符匹配（globbing）：是指使用通配符进行模式匹配的过程。通配符的正式名称叫做元字符通配符（metacharacter wildcard）。

创建文件 `touch`：创建文件并将你的用户名作为该文件的属主。

> `touch` 可以改变文件的修改时间为当前时间，而且不会改变文件内容。即对一个非空文件夹 `touch dir_name` 并不会改变文件夹 `dir_name` 里面的内容。

复制文件：基本用法 `cp source destination`

- 当想要在文件存在时 shell 询问「是否覆盖」，可以加上 `-i` 命令。通过 `y` 和 `n` 来回答；
- 用 `-R` 可以递归地复制整个目录的内容；



## Linux 基本命令



**创建文件夹**：`mkdir` 可以创建一级目录，如果要创建更深的目录（碰到不存在的文件夹自动创建可以使用 `mkdir -p`）

**查看权限**：

- 查看文件权限：`ls -l`
- 查看所在文件夹权限：`ls -ld`
- 修改文件夹权限：`chmod xxx dir/file` 其中 xxx 不同的数字代表不同的权限
  - 600 只有所有者有读和写的权限；644 所有者有读和写的权限，组用户只有读的权限；700 只有所有者有读和写以及执行的权限；666 每个人都有读和写的权限；777 每个人都有读和写以及执行的权限；

**移动文件或者目录**：`mv 原路径 目标路径`

**复制文件**：`cp 原路径 目标路径`

查看 CUDA 和 cuDNN 的版本：

1、方法一：查看 CUDA 版本 `nvcc --version` 或 `nvcc -V`；方法二：`cat /usr/local/cuda/version.json`

2、查看 cuDNN 版本：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

**修改 `/tmp` 目录**：

- 仅对当前终端有效：··

```shell
$ mkdir -p $HOME/tmp  # 在 HOME 目录创建一个 tmp 文件夹
$ export TMPDIR=$HOME/tmp  # 指定
```

- 永久生效：

```shell
$ mkdir -p $HOME/tmp
$ echo "export TMPDIR=$HOME/tmp" >> ~/.bashrc
$ source ~/.bashrc
```

`ncdu`

**查看文件大小**：https://blog.51cto.com/u_14691718/3432088

创建软连接：

查看已经挂载的硬盘：`df -T`，其中 `-T` 表示显示文件类型

查看磁盘占用情况：`df -h`

查看所有被系统识别的磁盘：`fdisk -l`

挂载 device 到 dir：`mount -t <type> <device> <dir>`



## 终端复用器 Tmux

> 推荐阅读：
>
> - [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

Tmux 是一个终端复用器（terminal multiplexer），在不同系统上安装的方式如下：

```shell
# Ubuntu 或 Debian
$ sudo apt-get install tmux

# CentOS 或 Fedora
$ sudo yum install tmux

# Mac
$ brew install tmux
```

输入 `tmux` 就可以开始使用，输入 `exit` 或者 `Ctrl + d` 就可以退出 Tmux 窗口：

**新建一个指定名字的会话**：`tmux new -s <session-name>`

**重新接入某个已存在的会话**：`tmux attach -t <session-name>`

**将当前会话与窗口分离**：`tmux detach` 或者 快捷键 `Ctrl + b, d`

**查看当前所有的 Tmux 会话**：`tmux ls` 或者 `tmux attach -t <session-name>`

**关掉某个会话**：`tmux kill-session -t 0` 或者 `tmux kill-session -t <session-name>`

**切换会话**：`tmux switch -t <session-name>`

**重命名会话**：`tmux rename-session -t 0 <new-name>`

**重命名当前窗口**：`tmux rename-window <new-name>`

## 段错误

https://komodor.com/learn/sigsegv-segmentation-faults-signal-11-exit-code-139/

https://tldp.org/FAQ/sig11/html/index.html

https://stackoverflow.com/questions/26521401/segmentation-fault-signal-11

https://github.com/JuliaLang/julia/issues/35005

https://discourse.julialang.org/t/signal-11-segmentation-fault-11-with-differentialequations-jl/23264

https://github.com/JuliaLang/julia/issues/31758
