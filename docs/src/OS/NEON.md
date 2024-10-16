NEON Instrinsic：

数据类型命名遵循规则：`<基本类型>x<lane 个数>x<向量个数>_t`

基本类型有 `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float64`, `float32`

lane 个数表示并行处理的基本类型数据的个数。

指令函数的命名：

常用 NEON 指令集命名遵循规则：`v<指令名>[后缀]_<数据基本类型简写>`

1. 如果没有后缀，表示 64 位并行；
2. 如果后缀是 q 表示 128 位并行；
3. 如果后缀是 l 表示长指令，输出类型的基本类型位数是输入的 2 倍；
4. 如果后缀是 n 表示窄指令，输出数据的基本类型位数是输入的一半；

数据基本类型简写：`s8`, `s16`, `s32`, `s64`, `u8`, `u16`, `u32`, `u64`, `f16`, `f32`

例如：

`vadd_u16`：两个 `uint16x4` 相加为一个 `uint16x4`

`vaddq_u16`：两个 `uint16x8` 相加为一个 `uint16x8`

`vaddl_u16`：两个 `uint8x8` 相加为一个 `uint16x8`

NEON 指令名分为：算术和位运算指令、数据移动指令、访存指令

算术和位运算指令包括 `add`, `sub`, `mul`

`dup[后缀]n<数据基本类型简写>`：用同一个标量值初始化一个向量全部的 lane；

`set[后缀]lane<数据基本类型简写>`：对指定的一个 lane 进行设置

`get[后缀]lane<数据基本类型简写>`：获取指定的一个 lane 的值

`mov[后缀]_<数据基本类型简写>`：数据间移动

`ld<向量数>[后缀]_<数据基本类型简写>`：读内存

`st<向量数>[后缀]_<数据基本类型简写>`：写内存

参考：

1. [arm Developer - Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
2. [neon 指令速查 2](https://blog.csdn.net/billbliss/article/details/78924636)

