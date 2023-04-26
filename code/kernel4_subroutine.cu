__device__ void warpReduce(volatile int * temp, int tIdx) {
    temp[tIdx] += temp[tIdx + 32];
    temp[tIdx] += temp[tIdx + 16];
    temp[tIdx] += temp[tIdx + 8];
    temp[tIdx] += temp[tIdx + 4];
    temp[tIdx] += temp[tIdx + 2];
    temp[tIdx] += temp[tIdx + 1];
}
