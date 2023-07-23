#include <stdio.h>

__global__ void my_first_kernel()
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    printf("Hello World from thread(thread index:(%d,%d), block index:(%d,%d))!\n", tidy, tidx, bidy, bidx);
}

// thread  --> block --> grid
// SM stream multi-processor
// total threads: block_size * grid_size
int main()
{
    printf("Hello World from CPU!\n");

    dim3 block_size(3,3);
    dim3 grid_size(2,2);

    my_first_kernel<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();

    return 0;
}
