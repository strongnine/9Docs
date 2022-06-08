## Linux 命令

**创建文件夹**：`mkdir` 可以创建一级目录，如果要创建更深的目录（碰到不存在的文件夹自动创建可以使用 `mkdir -p`）

**查看权限**：

- 查看文件权限：`ls -l`
- 查看所在文件夹权限：`ls -ld`
- 修改文件夹权限：`chmod xxx dir/file` 其中 xxx 不同的数字代表不同的权限
  - 600 只有所有者有读和写的权限；644 所有者有读和写的权限，组用户只有读的权限；700 只有所有者有读和写以及执行的权限；666 每个人都有读和写的权限；777 每个人都有读和写以及执行的权限；

**移动文件或者目录**：`mv 原路径 目标路径`

查看 CUDA 和 cuDNN 的版本：

1、方法一：查看 CUDA 版本 `nvcc --version` 或 `nvcc -V`；方法二：`cat /usr/local/cuda/version.json`

2、查看 cuDNN 版本：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

**修改 `/tmp` 目录**：

- 仅对当前终端有效：

```shell
$ mkdir -p $HOME/tmp  # 在 HOME 目录创建一个 tmp 文件夹
$ export TMPDIR=$HOME/tmp  # 指定
```

