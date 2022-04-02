# 做题记录

参考网站：

- [LeetCode 中国](https://leetcode-cn.com/)：最大最全的题库网站；
- [CodeTop 企业题库](https://codetop.cc/home)：能够根据不同的公司选择题库，有专门性地刷；
- [labuladong 的算法小抄](https://labuladong.github.io/algo/)
- ……

## 心路历程

- 刷题也好，面试也好，都是有技巧的；
- 数据结构是工具，算法是通过合适的工具解决特定问题的方法。—— labuladong；
- 做题先做「二叉树」，最容易培养框架思维，大部分算法技巧本质上都是树的遍历问题。—— labuladong；



## 二分查找（Binary Search）

[二分查找](https://strongnine.github.io/9Docs/dev/leetcode/algorithm/#%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE%EF%BC%88Binary%20Search%EF%BC%89)的题目，就看 liweiwei 的题解就行了：[写对二分查找不能靠模板，需要理解加练习 （附练习题）](https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/)

一般情况下，边界设置为 `left = mid + 1` 与 `right = mid`，这个时候中点是下取整，即偏向于左边取：`mid = (right - left) / 2 + left`。

> 当看到边界设置行为是 `left = mid` 与 `right = mid - 1` 的时候，需要将 `mid = (right - left + 1) / 2 + left`，即调整为上取整，即偏向于右边取。

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/liang-ci-er-fen-cha-zhao-by-strongnine-9-04l4/)；

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-rmzn/)；

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-toku/)；

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-k84i/)；

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-mszd/)；

[69. Sqrt(x)](https://leetcode-cn.com/problems/sqrtx/)，简单题，字节算法，

## 快慢指针

## 左右指针

## 滑动窗口

## 前缀和数组

## 差分数组


## 动态规划（Dynamic Programming）

动态规划的 liweiwei 有一个关于买卖股票问题的题解：[暴力解法、动态规划（Java）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/bao-li-mei-ju-dong-tai-gui-hua-chai-fen-si-xiang-b/)，还有[股票问题系列通解（转载翻译）](https://leetcode-cn.com/circle/article/qiAgHn/)。

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)，简单题，我的[题解](https://leetcode-cn.com/problems/maximum-subarray/solution/53-zui-da-zi-xu-he-dong-tai-gui-hua-by-s-csae/)；

**经典股票系列问题**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)，简单题，我的[题解](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/121-mai-mai-gu-piao-de-zui-jia-shi-ji-ji-54ir/)；

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)，中等题；

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)，困难题；

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)，困难题；

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)，中等题；

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)，中等题；

## 哈希（Hash）

[350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)，简单题，我的[题解](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/solution/350-liang-ge-shu-zu-de-jiao-ji-shi-yong-nyhsl/)；
