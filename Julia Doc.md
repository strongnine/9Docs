## Julia Documenter.jl 的使用

把修改 `push` 到仓库之前，可以先在本地看看能不能编译成功：

- `julia docs/make.jl` 没有报错就编译成功；编译成功会在 `docs` 目录里面出现一个 `build` 文件；
- `python3 -m http.server --bind localhost` 可以开启一个本地的网页服务器（部署在本地）；
- 打开 http://[::1]:8000/docs/build/ 可以访问，看看效果；

## Markdown 语法的一些注意点

外显的 LaTeX 公式不能够用 Typora 的语法，而是要另起一行用**内联公式**的语法（推荐用这种方法）。或者是用代码块（不推荐，因为在 Typora 中代码块的 LaTeX 公式不能可视化），语言设定为 `math`。

> （1）记住用公式块的时候，dollar 号（\$）前面不要有空格，否则会报错；
>
> （2）数学公式不能够作为一段的开头，即第一段的开头字符不能是 \$，否则这个公式会被变成单独显示

## 报错解决

最常碰到的错误是 `ERROR: LoadError: MethodError: no method matching mdflatten`，这个错误应该某些 Markdown 语法有错误，导致到无法识别。

当执行 `julia docs/make.jl` 的时候，会把放在 `docs` 的所有 md 文件都编译成一个 html 文件，然后再根据 make.jl 中指定的文件创建网页引导，所有的页面文件都存在 build 文件夹中。

遇到这个报错的解决方法就是，一个一个地把最近修改过的文件移出 docs 文件夹，尝试运行 `julia docs/make.jl`，直到不再报错，那么就是那个被移出去的文件中有错误。

