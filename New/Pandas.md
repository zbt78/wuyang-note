## Maps

```python
map()

# 求平均数
reviews_points_mean = reviews.points.mean()
# 让没一个point减去mean，这样得到的新的数据的平均数就是0
reviews.points.map(lambda p: p - review_points_mean)
```
---
```python
apply()
/**
** args:
		axis: {0 or 'index', 1 or 'columns'},默认为0，
			0 or 'index': apply function to each column 
			1 or 'columns': apply function to each row
	# apply()对 DataFrame的每一列或者每一行应用该函数，行/列由axis指定
**/
def remean_points(row):
	row.points = row.points - review_points_mean
	return row

reviews.apply(remean_points, axis = 'columns')

```

>map() and apply() return new data, but they don't modify the original data.
---
```python
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean
# 这段代码也能产生和上面相同的效果，并且速度更快
```
---
## 统计函数

```python
# 某列中出现的值，不包含重复的数据
unique()
# 某列中每个值出现的次数 
value_counts()

```