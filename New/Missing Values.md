## Three Approaches

#### 1. A Simple option is to drop columns with missing values
```python
# 删除含有 NA 的列
cols_with_missing = [col for col in X_train.columns
                        if X_train[col].isnull().any()]

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis = 1)
```

---
#### 2. A Better Option: Imputation. 

For instance, fill in the mean value
```python
# 使用 中位数 填充
my_imputer = SimpleImputer(strategy = 'median') 
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

print(imputed_X_train.columns)
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print(imputed_X_train.columns)
```
---
#### 3. An Extension To Imputation
![[Pasted image 20221025154657.png]]
