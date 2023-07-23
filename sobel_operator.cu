
#include <opencv2/opencv.hpp>
#include <iostream>
#include "error.cuh"

using namespace std;
using namespace cv;

__global__ void sobel_gpu(unsigned char* in, unsigned char* out, const int Height, const int Width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int index = y * Width + x;

    int Gx =0;
    int Gy =0;
    unsigned char x0, x1, x2, x3, x5, x6, x7, x8;
    //SM register
    if( x > 0 && x < (Width -1) && y>0 && y <(Height-1) )
    {
        x0 = in[(y-1)*Width + (x-1)];
        x1 = in[(y-1)*Width + (x)];
        x2 = in[(y-1)*Width + (x+1)];
        x3 = in[(y)*Width + (x-1)];
        //x4 = in[y * Width +  x];
        x5 = in[(y)*Width + (x+1)];
        x6 = in[(y+1)*Width + (x-1)];
        x7 = in[(y+1)*Width + (x)];
        x8 = in[(y+1)*Width + (x+1)];

        Gx = (x0 + 2*x3 +x6) - (x2 + 2*x5 + x8);
        Gy = (x0 + 2*x1 +x2) - (x6 + 2*x7 + x8);

        out[index] = (abs(Gx) + abs(Gy))/2;
    }

}

int main()
{
    Mat img = imread("1.jpg", 0);
    int height = img.rows;
    int width  = img.cols;

    Mat gaussImg;
    GaussianBlur(img, gaussImg, Size(3,3), 0, 0, BORDER_DEFAULT);

    Mat dst_gpu(height, width, CV_8UC1, Scalar(0));

    int memsize = height * width *sizeof(unsigned char);

    unsigned char* in_gpu;
    unsigned char* out_gpu;

    cudaMalloc((void**)&in_gpu, memsize);
    cudaMalloc((void**)&out_gpu, memsize);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x -1)/threadsPerBlock.x, (height + threadsPerBlock.y -1)/threadsPerBlock.y);
    
    cudaMemcpy(in_gpu, gaussImg.data, memsize, cudaMemcpyHostToDevice);

    sobel_gpu<<<blocksPerGrid, threadsPerBlock>>>(in_gpu, out_gpu, height, width);
    cudaError_t error_code;
    error_code = cudaGetLastError();

    if(error_code != cudaSuccess)
    {
        printf("error_code: %s\n", cudaGetErrorString(error_code));
    }

    cudaMemcpy(dst_gpu.data, out_gpu, memsize, cudaMemcpyDeviceToHost);

    imwrite("save.png", dst_gpu);
    cudaFree(in_gpu);
    cudaFree(out_gpu);

    return 0;
}



