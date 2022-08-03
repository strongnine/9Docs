## Linux 命令

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
