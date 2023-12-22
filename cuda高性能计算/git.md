## git工作区、暂存区、版本库

- 工作区：在电脑能看到的目录
- 暂存区：stage或index。一般放在`.git`目录下的index文件
- 版本库：又名仓库，英文名`repository`，可以理解为一个目录，这个目录里面所有的文件都可以被git管理起来。

### 创建目录

使用当前目录为git仓库，只需要初始化：

```shell
git init
```

使用指定目录作为git仓库：

```shell
git init newtest
```

### 添加文件到暂存区

用以下命令把文件添加到暂存区：

```shell
git add readme.txt
git add . # 提交所有文件
```

### 提交到仓库

用以下命令把文件提交到仓库：

```shell
git commit -m "第一次提交"
```

## 版本回退

查看历史提交记录：

```shell
git log
git log --pretty=oneline
```

版本回退：

```shell
git reset --hard HEAD^
# 或
git reset --hard 'commit id'
```

> 在git中，用`HEAD`表示当前版本，上一个版本就是`HEAD^`，上上一个版本就是`HEAD^^`，往上一百个版本可以写成`HEAD~100`。

使用一下命令查看历史，可以得知每次的`commit id`：

```shell
git reflog
```

## 查看仓库当前状态

查看状态：

```shell
git status 
```



![image-20230523104812485](https://blog-img-zbt.oss-cn-beijing.aliyuncs.com/picture/wuyang/202305231048805.png)

readme.txt 文件没有被添加，所以它的状态是`Untraacked`。

现在把readme.txt添加到暂存区并且提交，状态为`working tree clean`：

![image-20230523105139978](https://blog-img-zbt.oss-cn-beijing.aliyuncs.com/picture/wuyang/202305231051021.png)

## 远程仓库

### 添加远程仓库-Github

首先创建仓库，然后找到github的仓库地址：

![image-20230523105333111](https://blog-img-zbt.oss-cn-beijing.aliyuncs.com/picture/wuyang/202305231053151.png)

执行命令：

```shell
git remote add origin https://github.com/zbt78/test.git
```

添加后，远程仓库的名字就是origin，也可以改成别的。

下一步，把本地库的所有内容推送到远程库上：

```shell
git push -u origin master
```

### 移除远程仓库

查看远程库信息：

```shell
git remote -v
```

可以根据名字删除：

```shell
git remote rm origin
```

## 分支管理

### 基本操作

首先，列出分支：

```shell
git branch
```

创建分支*testing*：

```shell
git branch testing
```

切换分支：

```shell
git checkout testing
```

删除分支：

```shell
git branch -d testing
```

推送所有本地分支:

```bash
git branch --all
```

设置默认分支:

github主页 ->  Setitings -> Branches



删除远端分支:

git push {远程库remote名称} -d {远程分支名称}



### 合并冲突

yours  **->**  results   <- remote

提交冲突的原因:

提交者的版本库 < 远程库

如何实现同步: 

```bash
git pull
```





### 提交PR👻

- 点击**fork**push到自己的仓库
- git clone 到本地
-  

#### fork

- fork不是Git操作, 而是一个GitHub操作,
- fork后会在自己的github账户创建一个新仓库,包含原来仓库的所有内容

## PR流程:

- git commit .
- 格式化文档：clang-format -i -style=file paddle/phi/kernels/gpu/lerp_kernel.cu

- 
