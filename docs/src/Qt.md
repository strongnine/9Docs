## 问题总结

### 「欢乐斗地主」项目

课程地址：[大丙课程](https://edu.subingwen.cn/detail/p_619f2ad6e4b09240f0e3f719/6?product_id=p_619f2ad6e4b09240f0e3f719)；用手机登录；

Bug：`Undefined symbols for architecture arm64`，完整的报错信息：

设备：MacBook Pro Apple M1

系统：macOS Ventura 13.0

Qt Creator：**Qt Creator 9.0.0** Based on Qt 6.4.1 (Clang 13.0 (Apple), arm64)

Qt：Qt 6.4.0

```bash
Undefined symbols for architecture arm64:
  "Player::prepareCallLord()", referenced from:
      vtable for Player in moc_player.o
  "Player::preparePlayHand()", referenced from:
      vtable for Player in moc_player.o
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [Landlords.app/Contents/MacOS/Landlords] Error 1
23:00:12: The process "/usr/bin/make" exited with code 2.
Error while building/deploying project Landlords (kit: Desktop (arm-darwin-generic-mach_o-64bit))
When executing step "Make"
```

**尝试 1**：有可能是因为 Qt 版本的原因，项目视频里面是用 5.15.2 的版本。

先试着链接 https://www.cnblogs.com/wqcwood/p/15138983.html 中的方法：

做到 `make -j15` 这一步的时候报错了；

**尝试 2**：使用 BrewHome 安装了 Qt 5.15.7，设置 Qt mkspec 路径 `/opt/homebrew/opt/qt@5/mkspecs/linux-g++`，编译的时候出现报错：

```shell
make: *** [scorepanel.o] Error 1
make: *** Waiting for unfinished jobs....
In file included from ../Landlords/userplayer.cpp:1:
In file included from ../Landlords/userplayer.h:4:
In file included from ../Landlords/player.h:4:
In file included from /opt/homebrew/Cellar/qt@5/5.15.7/lib/QtCore.framework/Headers/QObject:1:
/opt/homebrew/Cellar/qt@5/5.15.7/lib/QtCore.framework/Headers/qobject.h:46:10: fatal error: 'QtCore/qobjectdefs.h' file not found
#include <QtCore/qobjectdefs.h>
         ^~~~~~~~~~~~~~~~~~~~~~
1 error generated.
1 error generated.
make: *** [cardpanel.o] Error 1
make: *** [main.o] Error 1
1 error generated.
make: *** [robot.o] Error 1
1 error generated.
make: *** [player.o] Error 1
1 error generated.
make: *** [userplayer.o] Error 1
1 error generated.
make: *** [gamecontrol.o] Error 1
11:46:07: The process "/usr/bin/make" exited with code 2.
Error while building/deploying project Landlords (kit: Desktop (arm-darwin-generic-mach_o-64bit))
When executing step "Make"
```

**尝试 3**：应该是虚函数没有定义。

如果是基类声明了一个虚函数，但是没有为其定义函数体，那么就会出现这个错误；

因此有两个修改的方式：

- 第一个方法：就是将 `Player.h` 头文件中的虚函数写成纯虚函数 `virtual void prepareCallLord() = 0;`，只需要在后面令虚函数等于 0 就是纯虚函数；
- 第二个方法：在 `Player.cpp` 源文件中为虚函数定义一个空的函数体；

❓为什么 Qt Creator 里「Design（设计）」的按钮是灰色的？

如果选中的文件不是一个可以使用 Design 的类型的，那么就无法进行设计。一般是后缀为 `ui` 的文件可以，放在 `Forms` 目录里面，双击选中这个文件就会跳出 Design 的界面了。