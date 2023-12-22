```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
```
---
#### 1. Drop Categorical Variables

只有在这列不包含有用的信息时才有效

> select_dtypes()
```python
# DataFrame.select_dtypes(_include=None_, _exclude=None_)
# Return a subset of the DataFrame’s columns based on the column dtypes.
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```

---

#### 2. Ordinal Encoding（序数编码）

![[Pasted image 20221025170620.png]]
把每个独特的值映射成不同的整数
```python
# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```

>sklearn.preprocessing.OrdinalEncoder
```python
# sklearn.preprocessing.OrdinalEncoder
# Encode categorical features as an integer array.
 
>>> from sklearn.preprocessing import OrdinalEncoder
>>> enc = OrdinalEncoder()
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OrdinalEncoder()
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 3], ['Male', 1]])
array([[0., 2.],
       [1., 0.]])
# ['Female', 3]和['Male', 1]都属于X
# ['Female', 3] -> [0., 2.]   Female -> 0 ;  3 -> 2
# ['Male', 1] -> [1., 0.]    Male -> 1 ; 1 -> 0
```
---

#### 3. One-Hot Encoding（热独编码）

![[Pasted image 20221025170914.png]]

```python 
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```


