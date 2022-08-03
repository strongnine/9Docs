### Docker 命令

> 推荐阅读：
>
> - [Docker 命令大全](https://www.runoob.com/docker/docker-command-manual.html)；

**docker run**：命令是创建一个新的容器并且运行一个命令，完整的语法为

`docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`

其中 OPTION 可以选参数：

`-i` 以交互模式运行容器，通常与 `-t` 同时使用；

`-t` 为容器分配一个伪输入终端，通常与 `-t` 同时使用；

> 通常都是 `-it` 这样的

`--volume, -v` 是绑定一个卷；或者说是目录映射；`-v 本地目录:容器目录`

`-e` 是环境变量；

> 例如 `-v /data:/data` 就是主机的目录 `/data` 映射到容器的 `/data`；

`--env-file=[]` 从指定文件读入环境变量；

`--ipc=host` 是容器间都共享宿主机的内存；

**登录**：`docker login 仓库地址 -u 用户名 -p 密码`（如果不指定地址则为登录官方仓库）

**登出**：`docker logout`

**拉取镜像**：`docker pull 镜像仓库地址`

**上传镜像**：`docker push 镜像仓库地址`

**列出容器**：`docker ps`
