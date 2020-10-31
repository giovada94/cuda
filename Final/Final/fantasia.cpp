#include "fantasia.h"
#include "MyImg.h"


std::string getExePath() {
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");
    return std::string(buffer).substr(0, pos);
}

int resultCudaAlloc(cudaError_t cudaStatus,std::string msg) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, msg.c_str());
        std::cout << std::endl;
        return 1;
    }
    else {
        return 0;
    }
}

void confrontoW(int *histogram,  float *sigma) {
    



    std::cout << std::endl << "HISTOGRAM" << std::endl;
   /* for (int i = 0;i < MAX_INT;i++) {
        if (histogram[i] != my_histogram[i]) {
            std::cout << i << "     " << histogram[i] << "    " << my_histogram[i] <<std::endl;
        }
    }*/

    
    
    
    std::cout << " W1 W2 " << std::endl;
    int w1, w2;
    float u1, u2;
    float my_sigma[256];
    for (int t = 0;t < MAX_INT;t++) {
        w1 = 0;
        w2 = 0;
        u1 = 0;
        u2 = 0;
        for (int i = 0;i < MAX_INT;i++) {
            if (t >= i) {
                w1+=histogram[i];
            }
            else {
                w2 += histogram[i];
            }
        }
        for (int i = 0;i < MAX_INT;i++) {
            if (t >= i) {
                u1 += (float)(i * histogram[i]) / w1;
            }
            else {
                u2 += (float)(i * histogram[i]) / w2;
            }
        }
        my_sigma[t] = (float)w1 * w2 * abs((u1 - u2)) * abs((u1 - u2));
        if (my_sigma[t] < 0) {
            std::cout<< my_sigma[t]<<" " << w1 << " "<< w2 << " " <<u1 << " "<< u2<<std::endl;
        }
    }

    for (int i = 0;i < MAX_INT;i++) {
        if (sigma[i] != my_sigma[i]) {
           // std::cout <<i<<": "<< sigma[i] << "   " << my_sigma[i] << std::endl;
        }
    }
}

void printBinImgByArray(int rows,int cols,int * array_bin) {
    cv::Mat src_tm = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // std::cout<<(int)src.at<uchar>(i, j)<<std::endl;
            if (array_bin[cols * i + j] == -1) {
                src_tm.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }


        }
    }
    cv::resize(src_tm, src_tm, cv::Size(cols * 0.75, rows * 0.75), 0, 0, cv::INTER_LINEAR);
    cv::imshow("image", src_tm);

    cv::waitKey(0);
}
