temp[threadIdx.x] = 
    d_in[threadIdx.x + 2*blockIdx.x*blockDim.x] +
    d_in[threadIdx.x + 2*blockIdx.x*blockDim.x + blockDim.x];

