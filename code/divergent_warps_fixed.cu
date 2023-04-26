__global__ void good_kernel() {
    if (threadIdx.x == 0) do_A();
    if (threadIdx.x == 32) do_B();
    if (threadIdx.x == 64) do_C();
    if (threadIdx.x == 96) do_D();
}
