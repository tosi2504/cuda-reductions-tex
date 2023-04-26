// do treereduction in sequential addressing style
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)  {
    if (threadIdx.x < stride) {
        temp[threadIdx.x] += temp[threadIdx.x + stride];
    }
    __syncthreads();
}


