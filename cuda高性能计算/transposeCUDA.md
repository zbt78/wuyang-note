```c++
__global__ void main_transpose(int m, int n, float* pweight, float* pweight_transpose) {
        __shared__ float local_memory[256];

        int gx = threadIdx.x + blockDim.x * blockIdx.x;
        int gy = threadIdx.y + blockDim.y * blockIdx.y;

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        if(gx < n && gy < m) {
                *(local_memory + ty * 16 + tx) = *(pweight + gy * n + gx);
        }

        __syncthreads();

        int x_trans = threadIdx.x + blockDim.y * blockIdx.y;
        int y_trans = threadIdx.y + blockDim.x * blockIdx.x;

        if(x_trans < m && y_trans < n) {
                *(pweight_transpose + y_trans * m + x_trans) = *(local_memory + tx * 16 + ty);
        }

        __syncthreads();
}

```

