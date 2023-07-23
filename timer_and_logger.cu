#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 32

void cpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    for(int y=0; y<size; ++y)
    {
        for(int x=0; x<size; ++x)
        {
            int tmp = 0;
            for(int step = 0; step < size; ++step)
            {
                tmp += a[y*size + step] * b[step * size + x];
            }
            c[y * size + x] = tmp;
        }
    }
}

__global__ void gpu_matrix_mult(int *a, int *b, int *c, const int size)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int tmp = 0;
    if( x < size && y < size)
    {
        for( int step = 0; step < size; ++step)
        {
            tmp += a[y * size + step] * b[step * size + x];
        }
        c[y * size + x] = tmp;
    }
}



int main()
{
    int matrix_size = 1000;
    int memsize = sizeof(int) * matrix_size * matrix_size;


    int *h_a, *h_b, *h_c, *h_cc;

    cudaMallocHost( (void**)&h_a, memsize);
    cudaMallocHost( (void**)&h_b, memsize);
    cudaMallocHost( (void**)&h_c, memsize);
    cudaMallocHost( (void**)&h_cc, memsize);

    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            h_a[y * matrix_size + x] = rand() % 1024;
        }
    }

    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            h_b[y * matrix_size + x] = rand() % 1024;
        }
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a , memsize);
    cudaMalloc((void**) &d_b , memsize);
    cudaMalloc((void**) &d_c , memsize);
    cudaError_t error_code;

    cudaEvent_t start, stop_cpu, stop_gpu;
    CHECK(cudaEventCreate(&start));
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);

    CHECK(cudaMemcpy( d_a, h_a, memsize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy( d_b, h_b, memsize, cudaMemcpyHostToDevice));


    unsigned int grid_rows = (matrix_size +BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size +BLOCK_SIZE -1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);//1.gpu warp 32 2. <= 1024

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, matrix_size);
    error_code = cudaGetLastError();
    if(error_code != cudaSuccess)
    {
        printf("Errors info: %s\n", cudaGetErrorString(error_code));
        printf("FILE: %s\n", __FILE__);
        printf("LINE: %d\n", __LINE__);
    }
    cudaMemcpy( h_c, d_c, memsize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_gpu);

    cpu_matrix_mult(h_a, h_b, h_cc, matrix_size);

    cudaEventRecord(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    printf("GPU time: %.2f ms\n", time_gpu);
    printf("CPU time: %.2f ms\n", time_cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);


    bool errors = false;
    for(int y=0; y<matrix_size; ++y)
    {
        for(int x=0; x<matrix_size; ++x)
        {
            if(fabs(h_cc[y*matrix_size + x] - h_c[y*matrix_size + x]) > (1.0e-10))
            {
                //printf("%d, %d\n", y, x);
                errors = true;
            }
        }
    }
    printf("Result: %s\n", errors?"Errors":"Passed");

    cudaFreeHost(h_a );
    cudaFreeHost(h_b );
    cudaFreeHost(h_c );
    cudaFreeHost(h_cc );
    cudaFree(d_a );
    cudaFree(d_b );
    cudaFree(d_c );
    return 0;

}


