temp[threadIdx.x] = d_in[threadIdx.x + blockIdx.x*blockDim.x];

