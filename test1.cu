#include "cuda_runtime.h"
#include "device_launch_parameter.h"
#include <cuda.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <device_functions.h>
#include <math.h>
using namespace std;
using namespace cv;

__global__ void sobelInCuda(unsigned char* dataIn, unsigned char* dataOut, int imgHeight, int imgWidth){
    int xIndex = threadIdx.x + blockDim.x + blockIdx.x;
    int yIndex = threadIdx.y + blockDim.y + blockIdx.y;
    int index = yIndex * imgWidth + xIndex;

    int Gx = 0;
    int Gy = 0;

    if(xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1){
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);

        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
        - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);

        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
    
}

int main(){
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera\n";
        return 1;
    }

    Mat srcImg;
    //namedWindow("live", WINDOW_AUTOSIZE); // 命名一個視窗，可不寫
    while (true) {
        // 擷取影像
        bool ret = cap.read(srcImg); // or cap >> frame;
        if (!ret) {
            cout << "Can't receive frame (stream end?). Exiting ...\n";
            break;
        }

        Mat gauImg;
        GaussianBlur(srcImg, gauImg, Size(3,3), 0, 0, 4);

        Mat grayImg;
        cvtColor(gauImg, grayImg, COLOR_BGR2GRAY);
        //namedWindow("gray", 1)
        //imshow("gray", grayImg);

        int imgHeight = grayImg.rows;
        int imgWidth = grayImg.cols;

        Mat dstGpuImg(imgHeight, imgWidth, CV_8UC1, Scalar(0,0,0));

        size_t num = imgHeight * imgWidth * sizeof(unsigned char);
        unsigned char* d_in;
        unsigned char* d_out;
        cudaMalloc((void**)&d_in, num);
        cudaMalloc((void**)&d_out, num);

        cudaMemcpy(d_in, grayImg.data, num, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(32,32);
        dim3 blockPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                            (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, 0);
        sobelInCuda<<<blockPerGrid, threadsPerBlock>>>(d_in, d_out, imgHeight, imgWidth);
        //checkCudaError("Kernel launch");
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);

        cudaMemcpy(dstGpuImg.data, d_out, num, cudaMemcpyDeviceToHost);
        

        namedWindow("Sobel", 1)
        imshow("Sobel", dstGpuImg);

        //cout << "elapsed time: " << elapsed_time << " ms" << endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        
        //checkCudaError("Memcpy Device to Host");

        cudaFree(d_in);
        cudaFree(d_out);
        //checkCudaError("cudaFree");

        waitKey(0);

    }
    destroyAllWindows();
    return 0;
}