// do treereduction in interleaved addressing style
for (int stride = 1; stride < blockDim.x; stride *= 2)  {
    if (threadIdx.x % (2*stride) == 0) {
        temp[threadIdx.x] += temp[threadIdx.x+stride];
    }
    __syncthreads();

}


