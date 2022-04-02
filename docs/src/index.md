# 9Docs

我是强劲九（strongnine），9Docs 是我的个人笔记文档，有问题也可以和我联系或者交流，微信「strongnine」，希望可以点击[这里](https://github.com/strongnine/9Docs)给这个文档一个 Star！

## 目录

- 数据结构与算法：
- 机器学习：
  - 深度学习；
  - 图神经网络；
- 产品思维：
  - 产品经理；
- 个人学习资料的总结和推荐：
  - [书籍推荐](https://strongnine.github.io/9Docs/dev/library/book)；
  - [课程推荐](https://strongnine.github.io/9Docs/dev/library/recommend )；

以下是我在不同平台的账号，欢迎关注：

GitHub：[strongnine](https://github.com/strongnine)；

公众号：strongnine；

CSDN：[strongnine](https://blog.csdn.net/weixin_39679367?spm=1001.2101.3001.5343)；

## 代表文章

- [将一维时间序列转化成二维图片](https://blog.csdn.net/weixin_39679367/article/details/86416439?spm=1001.2014.3001.5502)；
- [Python：使用 pyts 把一维时间序列转换成二维图片](https://blog.csdn.net/weixin_39679367/article/details/88653018?spm=1001.2014.3001.5502)（欢迎给[内附的 GitHub 代码](https://github.com/strongnine/Series2Image)点个 Star）；
- [LaTeX：公式常用字符和表达式](https://blog.csdn.net/weixin_39679367/article/details/84729452)；
- ......

## Julia Documenter.jl 的使用

把修改 `push` 到仓库之前，可以先在本地看看能不能编译成功：

- `julia docs/make.jl` 没有报错就编译成功；编译成功会在 `docs` 目录里面出现一个 `build` 文件；
- `python3 -m http.server --bind localhost` 可以开启一个本地的网页服务器（部署在本地）；
- 打开 http://[::1]:8000/docs/build/ 可以访问，看看效果；

