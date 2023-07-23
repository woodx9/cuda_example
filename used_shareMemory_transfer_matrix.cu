# include <stdio.h>
# include <math.h>

#define BLOCK_SIZE 32
#define M 3000
#define N 1000

__managed__ int matrix[N][M];
__managed__ int gpu_result[M][N];
__managed__ int cpu_result[M][N];

__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N])
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if( x < M && y < N)
    {
        out[x][y] = in[y][x];
    }
}

__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N])
{
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int ken[BLOCK_SIZE+1][BLOCK_SIZE+1];//ken[32] warp

    if(x < M && y < N)
    {
        ken[threadIdx.y][threadIdx.x] = in[y][x];
    }
    __syncthreads();

    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if(x1 < N && y1 < M)
    {
        out[y1][x1] = ken[threadIdx.x][threadIdx.y];//32 bank
    }

}

void cpu_matrix_transpose(int in[N][M], int out[M][N])
{
    for(int y = 0; y < N; y++)
    {
        for(int x = 0; x < M; x++)
        {
            out[x][y] = in[y][x];
        }
    }
}

int main()
{
    for(int y=0; y<N; y++)
    {
        for(int x=0; x<M; x++)
        {
            matrix[y][x] = rand()%1024;
        }
    }

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    dim3 dimGrid((M + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE -1)/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    for(int i = 0; i < 20; i++)
    {
        gpu_shared_matrix_transpose<<<dimGrid,dimBlock>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix_transpose(matrix, cpu_result);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_cpu, time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    bool errors = false;
    for(int y = 0; y<M; y++)
    {
        for (int x = 0; x < N; x++)
        {
            if(fabs(cpu_result[y][x] - gpu_result[y][x]) > (1.0e-10))
            {
                errors = true;
            }
        }
        
    }

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f\nGPU time: %.2f\n", time_cpu, time_gpu/20.0);

    return 0;
}







