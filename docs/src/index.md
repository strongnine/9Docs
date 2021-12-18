# 9Docs

我是强劲九，9Docs 是我的个人笔记文档，记录了在编程学习上的记录，目的是方便自己查看。

微信：strongnine；

公众号：strongnine；

## 学习历程

- [Git 学习笔记](https://strongnine.github.io/9Docs/dev/git/git_notebook)
- LeetCode 刷题笔记
    - [数据结构](https://strongnine.github.io/9Docs/dev/leetcode/data_structure)
    - [算法总结](https://strongnine.github.io/9Docs/dev/leetcode/algorithm)
    - [刷题记录](https://strongnine.github.io/9Docs/dev/leetcode/leetcoding)
	
- ......

## Julia Documenter.jl 的使用

把修改 `push` 到仓库之前，可以先在本地看看能不能编译成功：

- `julia docs/make.jl` 没有报错就编译成功；编译成功会在 `docs` 目录里面出现一个 `build` 文件；
- `python3 -m http.server --bind localhost` 可以开启一个本地的网页服务器（部署在本地）；
- 打开 http://[::1]:8000/docs/build/ 可以访问，看看效果；

