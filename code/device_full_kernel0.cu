__global__ void reduce(int * d_in, int * d_out) {

	// prepare shared data allocated by kernel invocation
    // and copy input array
	extern __shared__ int temp[];
	temp[threadIdx.x] = d_in[threadIdx.x + blockIdx.x*blockDim.x];
	__syncthreads();

	// do treereduction in interleaved addressing style
	for (int stride = 1; stride < blockDim.x; stride *= 2)  {
		if (threadIdx.x % (2*stride) == 0) {
			temp[threadIdx.x] += temp[threadIdx.x+stride];
		}
		__syncthreads();

	}

	// export result to global memory
	if (threadIdx.x == 0) {
		*d_out = temp[blockIdx.x];
	}
}
