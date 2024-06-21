
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        printf("Cannot open camera\n");
        return -1;
    }

    cv::Mat frame, gray, edges;
    char text[100];

    while (true) {
        clock_t start_time = clock();

        cap >> frame;

        if (frame.empty()) {
            printf("Can't receive frame (stream end?). Exiting ...\n");
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 100, 200);

        clock_t end_time = clock();
        double frame_time = double(end_time - start_time) / CLOCKS_PER_SEC;

        sprintf(text, "Time: %.3fs", frame_time);
        cv::putText(edges, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        // cv::imshow("Original", frame);
        cv::imshow("Canny Edges", edges);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
