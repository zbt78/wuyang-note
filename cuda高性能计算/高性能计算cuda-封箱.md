**终止于** chapter5共享内存填充: 

- 为啥填充一列后,按列主序访问就不会出现存储体冲突?

- 

- 线程束和存储体到底是怎么安排的?搞不懂呀.

- 这里为啥要 +1?

	```c++
	col_idx=threadIdx.x*(blockDim.x+1)+threadIdx.y
	```

- ![image-20230326133140805](https://blog-img-zbt.oss-cn-beijing.aliyuncs.com/picture/wuyang/202303261332002.png)