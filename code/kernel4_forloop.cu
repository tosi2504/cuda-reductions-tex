for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)  {
    if (threadIdx.x < stride) {
        temp[threadIdx.x] += temp[threadIdx.x + stride];
    }
    __syncthreads();
}

if (threadIdx.x < 32) warpReduce(temp, threadIdx.x);
