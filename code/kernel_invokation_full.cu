// set the number of threads per block and calculate the required number of blocks
int arraySize = sizeof(int)*len;
int numThreadsPerBlock = 1024;
int numBlocks = (arraySize + numThreadsPerBlock - 1) / numThreadsPerBlock;

// allocate and copy into device memory
int * d_in, * d_out;
cudaMalloc(&d_in, sizeof(int)*arraySize);
cudaMalloc(&d_out, sizeof(int)*numBlocks);
cudaMemcpy(d_in, h_in, sizeof(int)*arraySize, cudaMemcpyHostToDevice);

// invoke kernel with the correct amount of threads and cache space
reduce <<< numBlocks, numThreadsPerBlock, sizeof(int)*numThreadsPerBlock >>> (d_in, d_out);

// copy result to host and free memory
cudaMemcpy(h_out, d_out, sizeof(int)*numBlocks, cudaMemcpyDeviceToHost);
cudaFree(d_in);
cudaFree(d_out);
