// allocate and copy to memory of device
int arraySize = 1024;
int * d_in, * d_out;
cudaMalloc(&d_in, sizeof(int)*arraySize);
cudaMalloc(&d_out, sizeof(int));
cudaMemcpy(d_in, h_in, sizeof(int)*arraySize, cudaMemcpyHostToDevice);

// invoke kernel with the correct amount of threads and cache space
int numThreadsPerBlock = len;
int numBlocks = 1;
reduce <<< numBlocks, numThreadsPerBlock, sizeof(int)*numThreadsPerBlock >>> (d_in, d_out, len);

// copy result to host and free memory
cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(d_in);
cudaFree(d_out);
