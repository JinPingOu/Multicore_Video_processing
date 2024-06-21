#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "../cudafilters.hpp"
#include "../cudaarithm.hpp"
#include "../cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_frame, d_gray, d_blur, d_edges;
    cv::Mat edges;

    //cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(3,3), 1.2, 1.2);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(100.0, 200.0);

    while (true) {
        double start_time = (double)cv::getTickCount();

        cap >> edges; 
        if (edges.empty()) {
            std::cerr << "Can't receive frame (stream end?). Exiting ..." << std::endl;
            break;
        }

        d_frame.upload(edges);

        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY); 
        //cv::cuda::GaussianBlur(d_gray, d_blur, cv::Size(3, 3), 0); 
        canny->detect(d_blur, d_edges);

        d_edges.download(edges); 

        double end_time = (double)cv::getTickCount();
        double frame_time = (end_time - start_time) / cv::getTickFrequency();

        cv::putText(edges, cv::format("Time: %.3f s", frame_time), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Canny Edges", edges);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
