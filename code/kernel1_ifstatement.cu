// do treereduction in interleaved addressing style
for (int stride = 1; stride < blockDim.x; stride *= 2)  {
    int index = 2 * stride * threadIdx.x;
    if (index < blockDim.x) {
        temp[index] += temp[index + stride];
    }
    __syncthreads();
}


