# 做题记录

## 栈（stack）

[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)，简单题，我的[题解](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/solution/yong-liang-ge-zhan-shi-xian-dui-lie-by-s-0dtx/)；

[剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)，简单题，我的[题解](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/solution/wei-hu-liang-ge-zhan-lai-shi-xian-by-str-gyca/)；

[剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)，简单题，我的[题解](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/solution/san-chong-fang-fa-jie-jue-fan-xiang-da-y-irt5/)；

## 树（tree）

前序遍历、中序遍历、后序遍历基本写法：

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)，简单题，

[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)，简单题，

[145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)，简单题，

关于前中后序三种遍历，极客时间王争的课程给了三个对应的递推代码

```c++
// 前序遍历
void preOrder(Node* root) {
  if (root == null) return;
  print root // 此处为伪代码，表示打印root节点
  preOrder(root->left);
  preOrder(root->right);
}

// 中序遍历
void inOrder(Node* root) {
  if (root == null) return;
  inOrder(root->left);
  print root // 此处为伪代码，表示打印root节点
  inOrder(root->right);
}

// 后序遍历
void postOrder(Node* root) {
  if (root == null) return;
  postOrder(root->left);
  postOrder(root->right);
  print root // 此处为伪代码，表示打印root节点
}
```


## 链表（linked-list）

[剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)，我的[题解](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/die-dai-he-di-gui-liang-chong-fang-fa-by-s3su/)；

## 双指针

**简单题：**

[350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)，我的[题解](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/solution/350-liang-ge-shu-zu-de-jiao-ji-shi-yong-nyhsl/)；

**中等题：**

[82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)；



## 二分搜索（Binary Search）

二分查找的题目，就看 liweiwei 的题解就行了：[写对二分查找不能靠模板，需要理解加练习 （附练习题）](https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/)

一般情况下，边界设置为 `left = mid + 1` 与 `right = mid`，这个时候中点是下取整，即偏向于左边取：`mid = (right - left) / 2 + left`。

> 当看到边界设置行为是 `left = mid` 与 `right = mid - 1` 的时候，需要将 `mid = (right - left + 1) / 2 + left`，即调整为上取整，即偏向于右边取。

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/liang-ci-er-fen-cha-zhao-by-strongnine-9-04l4/)；

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-rmzn/)；

[81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-toku/)；

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-k84i/)；

[154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)，我的[题解](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/solution/er-fen-cha-zhao-de-lian-xi-by-strongnine-mszd/)；




## 动态规划（Dynamic Programming）

动态规划的 liweiwei 有一个关于买卖股票问题的题解：[暴力解法、动态规划（Java）](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/bao-li-mei-ju-dong-tai-gui-hua-chai-fen-si-xiang-b/)，还有[股票问题系列通解（转载翻译）](https://leetcode-cn.com/circle/article/qiAgHn/)。

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)，简单题，我的[题解](https://leetcode-cn.com/problems/maximum-subarray/solution/53-zui-da-zi-xu-he-dong-tai-gui-hua-by-s-csae/)；

**经典股票系列问题**

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)，简单题，我的[题解](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/solution/121-mai-mai-gu-piao-de-zui-jia-shi-ji-ji-54ir/)。

[122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)，中等题，

[123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)，困难题，

[188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)，困难题，

[309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)，中等题，

[714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)，中等题，

## 哈希（Hash）

[350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)，简单题，我的[题解](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/solution/350-liang-ge-shu-zu-de-jiao-ji-shi-yong-nyhsl/)；
