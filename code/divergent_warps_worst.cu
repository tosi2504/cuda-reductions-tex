__global__ void horrific_kernel() {
    if (threadIdx.x == 0) do_A();
    if (threadIdx.x == 1) do_B();
    if (threadIdx.x == 2) do_C();
    if (threadIdx.x == 3) do_D();
}
