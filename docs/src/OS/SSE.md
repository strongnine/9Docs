## 算法优化

### 指令优化

#### SSE 指令

SIMD（Single Instruction Multiple Data）即单指令多数据技术，目前 Intel 处理器支持的 SIMD 技术包括 MMX（Multi Media eXtension，多媒体扩展指令集）、SSE（Stream SIMD Extentions，数据流单指令多数据扩展）、AVX。这些指令的区别在 SIMD 操作的寄存器上，MMX、SSE、AVX 提供了 8 个寄存器做 SIMD 操作，大小分别为 64bit、128bit、256bit。

```c++
#include <mmintrin.h>   // MMX
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>  // SSE4.1
#include <nmmintrin.h>  // SSE4.2
#include <wmmintrin.h>  // AES and PCLMULQDQ intrinsics
#include <immintrin.h>  // Intel-specific intrinsics (AVX)
#include <ammintrin.h>  // AMD-specific intrinsics (FMA4, LWP, XOP)
#include <mm3dnow.h>    // AMD 3DNow! intrinsics
```

编译选项 `-msseN`, `-mavxN` 其中 N 表示版本编号。

数据位宽的名称包括三个部分：

1. 前缀 `__m`
2. 中间是数据位宽，指的是 SIMD 寄存器的位宽
3. 最后加上数据类型，`i` 为整数，`d` 为双精度浮点数，不加字母是单精度浮点数

SSE 指令集为 `__m128`, `__m128i`, `__m128d`

AVX 指令集为 `__m256`, `__m256i`, `__m256d`

intrinsic 函数命名由三个部分组成：

1. 前缀：MMX 和 SSE 都为 `__mm`，AVX 和 AVX-512 则会额外加上 256 和 512 的位宽标识
2. 执行操作：比如 `_add`, `_mul`, `_load`，`_loadu` 表示以无需内存对齐的方式加载数据
3. 数据范围和类型：`_ps` 的 `p` 表示 packed，对所有数据做操作，`s` 表示单精度浮点数；`_ss` 的 `s` 表示 single，只对第一个数据操作，`s` 同样表示单精度浮点。后面还会跟上位宽，例如 `_epi16`

intrinsic 函数可以在 [Intel Intrinsic Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) 网站中找到