读取文件和保存文件：[Input/Output](https://pandas.pydata.org/docs/reference/io.html)

## 通用函数

### 数据操作
| 函数 | 功能 |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| [`melt`](https://pandas.pydata.org/docs/reference/api/pandas.melt.html#pandas.melt)(frame[, id_vars, value_vars, var_name, ...]) | 将 DataFrame 从宽格式转为长格式，可选择设置标识符。          |
| [`pivot`](https://pandas.pydata.org/docs/reference/api/pandas.pivot.html#pandas.pivot)(data, *[, index, columns, values]) | 返回由给定索引 / 列值组织的重塑 DataFrame。 |
| [`pivot_table`](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html#pandas.pivot_table)(data[, values, index, columns, ...]) | 创建一个电子表格样式的数据透视表作为 DataFrame。 |
| [`crosstab`](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html#pandas.crosstab)(index, columns[, values, rownames, ...]) | 计算两个（或更多）因子的简单交叉表。 |
| [`cut`](https://pandas.pydata.org/docs/reference/api/pandas.cut.html#pandas.cut)(x, bins[, right, labels, retbins, ...]) | 将值分类为离散间隔。                |
| [`qcut`](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html#pandas.qcut)(x, q[, labels, retbins, precision, ...]) | 基于分位数的离散化函数。          |
| [`merge`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html#pandas.merge)(left, right[, how, on, left_on, ...]) | 将 DataFrame 或命名的 Series 对象与数据库样式的连接合并。 |
| [`merge_ordered`](https://pandas.pydata.org/docs/reference/api/pandas.merge_ordered.html#pandas.merge_ordered)(left, right[, on, left_on, ...]) | 使用可选的填充/插值对有序数据执行合并。 |
| [`merge_asof`](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html#pandas.merge_asof)(left, right[, on, left_on, ...]) | 按关键距离执行合并。                   |
| [`concat`](https://pandas.pydata.org/docs/reference/api/pandas.concat.html#pandas.concat)(objs, *[, axis, join, ignore_index, ...]) | 沿特定轴连接 pandas 对象。 |
| [`get_dummies`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html#pandas.get_dummies)(data[, prefix, prefix_sep, ...]) | 将分类变量转换为虚拟/指标变量。 |
| [`from_dummies`](https://pandas.pydata.org/docs/reference/api/pandas.from_dummies.html#pandas.from_dummies)(data[, sep, default_category]) | 从虚拟变量的 `DataFrame` 创建一个分类的 `DataFrame`。 |
| [`factorize`](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html#pandas.factorize)(values[, sort, na_sentinel, ...]) | 将对象编码为枚举类型或分类变量。 |
| [`unique`](https://pandas.pydata.org/docs/reference/api/pandas.unique.html#pandas.unique)(values) | 根据哈希表返回唯一值。       |
| [`wide_to_long`](https://pandas.pydata.org/docs/reference/api/pandas.wide_to_long.html#pandas.wide_to_long)(df, stubnames, i, j[, sep, suffix]) | 将 DataFrame 从宽格式转为长格式。 |

### 顶层缺失数据

| [`isna`](https://pandas.pydata.org/docs/reference/api/pandas.isna.html#pandas.isna)(obj) | Detect missing values for an array-like object.     |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [`isnull`](https://pandas.pydata.org/docs/reference/api/pandas.isnull.html#pandas.isnull)(obj) | Detect missing values for an array-like object.     |
| [`notna`](https://pandas.pydata.org/docs/reference/api/pandas.notna.html#pandas.notna)(obj) | Detect non-missing values for an array-like object. |
| [`notnull`](https://pandas.pydata.org/docs/reference/api/pandas.notnull.html#pandas.notnull)(obj) | Detect non-missing values for an array-like object. |
