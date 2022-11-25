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

## Pandas 常用操作

读取 CSV 文件：`pd.DataFrame.from_csv("csv_file")` 或者 `pd.read_csv("csv_file")`；

读取 Excel 文件：`pd.read_excel("excel_file")`

将 DataFrame 保存为 CSV 文件：采用逗号作为分隔符，且不带索引：`df.to_csv("data.csv", sep=',', index=False)`；

获得基本的数据集特征信息：`df.info()`；

基本的数据集统计信息：`df.describe()`；

将 DataFrame 输出到一张表：`print(tabulate(print_table, headers=headers))`，当 `print_table` 是一个列表，其中列表元素还是新的列表，`headers` 为表头字符串组成的列表；

列出所有列的名字：`df.columns`；

删除缺失数据：`df.dropna(axis=0, how='any')` 返回一个 DataFrame，其中删除了包含任何 `NaN` 值的给定轴，选择 `how = 'all'` 会删除所有元素都是 `NaN` 的给定轴；

替换缺失数据：`df.replace(to_replace=None, value=None)`，使用 `value` 值替代 DataFrame 中的 `to_replace` 值，其中 `value` 和 `to_replace` 都需要我们赋予不同的值；

检查空值 `NaN`：`pd.isnull(object)` 即数值数组中的 `NaN` 和目标数组中的 `None/NaN`；

删除特征：`df.drop('feature_variable_name', axis=1)`，`axis` 为 0 表示选择行，1 表示选择列；

将目标类型转换为浮点型：`pd.to_numeric(df["feature_name"], errors='coerce')` 将目标类型转化为数值从而进一步执行计算，在这个案例中为字符串；

将 DataFrame 转换为 NumPy 数组：`df.as_matrix()`；

查看 DataFrame 的前 n 行：`df.head(n)`；

查看 DataFrame 的后 n 行：`df.tail(n)`；

通过特征名取数据：`df.loc[feature_name]`；

对 DataFrame 使用函数：`df["height"].apple(lambda: height: 2 * height)` 将 `height` 行的所有值乘上 2；

重命名行：`df.rename(column={df.columns[2]: 'size'}, inplace=True)` 将第三行重命名为 `size`；

取某一行的唯一实体：`df["name"].unique()` 取 `name` 行的唯一实体；

访问子 DataFrame：`new_df = df[["name", "size"]]` 抽取行 `name` 和 `size`；

总结数据信息：总和 `df.sum()`，最小值 `df.min()`，最大值 `df.max()`，最小值的索引位置 `df.idxmin()`，最大值的索引位置 `df.idxmax()`，统计数据 `df.describe()`，均值 `df.mean()`，中值 `df.median()`，相关系数 `df.corr()`，这些函数都可以对单独数据行进行；

给数据排序：`df.sort_values(ascending=False)`；

布尔型索引：`df[df["size"] == 5]` 选择 `size` 行中值等于 5 的行；

选特定的值：`df.loc([0], ['size'])` 选的 `size` 列第一行数据；

沿着一条轴将多个对象堆叠到一起：`pd.concat()`；

关系型数据库的连接方式：`pd.merge()` 根据一个或多个键将不同的 DataFrame 连接起来。针对同一个主键存在两张不同字段的表，根据主键整合到一张表里；

用于索引上的合并：`pd.join()`

纵向追加：`pd.append()`



转换字符串时间为标准时间：`pd.to_datetime(df['time'])`；

提取时间序列信息：`df['time'].year()`

加减时间：`df['time'] + pd.Timedelta(days=1)` 或者 `df['time'] - pd.to_date_time('2016-1-1')`

时间跨度计算
