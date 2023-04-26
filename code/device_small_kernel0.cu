__global__ void reduce(int * d_in, int * d_out, int len) {

	// prepare shared data allocated by kernel invocation
    // and copy input array
	extern __shared__ int temp[];
	temp[threadIdx.x] = d_in[threadIdx.x];
	__syncthreads();

	// do treereduction in interleaved addressing style
    int stride = 1;
	for (int step = 0; step < log2(len); step++)  {

		if (threadIdx.x % (2*stride) == 0) {
			temp[threadIdx.x] += temp[threadIdx.x+stride];
		}
		__syncthreads();
        stride *= 2;
	}

	// export result to global memory
	if (threadIdx.x == 0) {
		*d_out = temp[0];
	}
}
