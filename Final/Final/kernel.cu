
#include "fantasia.h"
#include "MyImg.h"
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdlib>

/*
#include <npp.h>
#include <nppdefs.h>
#include <nppcore.h>
#include <nppi.h>
#include <npps.h>
*/


__global__ void kernel_setHist(int* hist, int* array_img, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&hist[array_img[tid]], 1);
}



__global__ void kernelOtsu(int* hist, float* sigma,int last) {
    int i = threadIdx.x;
    int t = blockIdx.x;
    __shared__ int s_w1, s_w2;
    __shared__ float s_u1, s_u2;
    if (i == 0) {
        s_w1 = 0;
        s_w2 = 0;
        s_u1 = 0;
        s_u2 = 0;
    }
    __syncthreads();
    if (t >= i) {
        atomicAdd(&s_w1, hist[i]);
        
    }
    else {
        atomicAdd(&s_w2, hist[i]);
    }
    __syncthreads();

    if (t >= i) {
        atomicAdd(&s_u1, (float)(i * hist[i]) / s_w1);
    }
    else {
        atomicAdd(&s_u2, (float)(i * hist[i]) / s_w2);
    }
    __syncthreads();
    
    if (i == last) {
        sigma[t] = (float)s_w1 * s_w2 * pow((s_u1 - s_u2),2);
    }
}
__global__ void binImg(int* img, int t, int size,int val)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (img[i] > t) {
        img[i] = val;
    }
    else {
        img[i] = 1;
    }
}


__global__ void FindMax(float* sigma, float* result,int* pos, int step) {


    __shared__ float my_array[1024];
    __shared__ int part_pos[1024];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    part_pos[tid] = i;
    my_array[tid] = sigma[i];

    __syncthreads();

    int half = MAX_INT;

    //10-> logarithm bse to 1024
    for (int t = 1;t <= step;t++) {
        half = half / 2;
        if (tid < half) {
            if (my_array[tid] < my_array[tid + half]) {
                my_array[tid] = my_array[tid + half];
                part_pos[tid] = part_pos[tid + half];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *result = my_array[0];
        *pos=part_pos[0];
    }
}


int getThresholding(int rows, int cols, int* histogram, int* array_img, int * thresholding) {
   


    cudaError_t cudaStatus;
    int* dev_histogram;
    int* dev_array_img;

    int res = 0;
    int size = rows * cols;
    cudaStatus = cudaMalloc((void**)&dev_histogram, MAX_INT * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev histogram");
    cudaStatus = cudaMemcpy(dev_histogram, histogram, MAX_INT * sizeof(int), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia histogram da host a device");
    cudaStatus = cudaMalloc((void**)&dev_array_img, size * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev array img");
    cudaStatus = cudaMemcpy(dev_array_img, array_img, size * sizeof(int), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia array img da host a device");



    dim3 block(1024);
    dim3 grid((size + block.x - 1) / block.x);

    kernel_setHist<< < grid, block >> >(dev_histogram, dev_array_img, size);
   
    float *sigma = (float*)malloc(MAX_INT * sizeof(float));

    for (int i = 0;i < MAX_INT;i++) {
        sigma[i] = 0;
    }
    float* dev_sigma;

    cudaStatus = cudaMalloc((void**)&dev_sigma, MAX_INT * sizeof(float));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev sigma");
    cudaStatus = cudaMemcpy(dev_sigma, sigma, MAX_INT * sizeof(float), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia sigma da host a device");


    dim3 block_o(MAX_INT);
    dim3 grid_o(MAX_INT);

    kernelOtsu<< < grid_o, block_o >> >(dev_histogram, dev_sigma,MAX_INT-1);


    float* dev_result = NULL;
    float* result= (float*)malloc(sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_result, sizeof(float));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev result");

    int* dev_pos_max = NULL;
    int* pos_max = (int*)malloc(sizeof(int));


    cudaStatus = cudaMalloc((void**)&dev_pos_max,sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev pos max");



    FindMax << < 1, MAX_INT >> > (dev_sigma,dev_result, dev_pos_max, 8);
    cudaStatus = cudaMemcpy(result, dev_result,  sizeof(float), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "Errore copia result da device a host");

    cudaStatus = cudaMemcpy(pos_max, dev_pos_max, sizeof(int), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "Errore copia pos max da device a host");
    *thresholding = *pos_max;
    cudaFree(dev_histogram);
    cudaFree(dev_array_img);
    cudaFree(dev_sigma);
    cudaFree(dev_pos_max);
    cudaFree(dev_result);
    //std::cout <<std::endl<< "result: " << *result <<"    "<< *pos_max<< std::endl;
    
    /*
    cudaStatus = cudaMemcpy(histogram, dev_histogram, MAX_INT * sizeof(int), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "Errore copia histogram da device a host");
    /*
    cudaStatus = cudaMemcpy(sigma, dev_sigma, MAX_INT * sizeof(float), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "Errore copia sigma da device a host");
    */


/*
    cudaFree(dev_histogram);
    cudaFree(dev_array_img);
    cudaFree(dev_u1);
    cudaFree(dev_u2);
*/
    /*
   for (int i = 0;i < MAX_INT;i++) {
        std::cout << i << ": "<<histogram[i] <<std::endl;
    }*/
   
    //confrontoW(histogram, sigma);



    return res;
}

int getBinary(int img_size,int* array_img, int* array_img_bin,int* thresholding,int val) {
    dim3 block(1024);
    dim3 grid((img_size + block.x - 1) / block.x);
    cudaError_t cudaStatus;
    int* dev_array_img_bin;




    int res = 0;

    cudaStatus = cudaMalloc((void**)&dev_array_img_bin, img_size * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev img bin");
    cudaStatus = cudaMemcpy(dev_array_img_bin, array_img, img_size * sizeof(int), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia img bin da host a device");
    binImg << < grid, block >> > (dev_array_img_bin, *thresholding, img_size,val);




    cudaStatus = cudaMemcpy(array_img_bin, dev_array_img_bin, img_size * sizeof(int), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "Errore copia img bin da device a host");
    cudaFree(dev_array_img_bin);
    return res;
}

__global__ void correlation(int* tmplt, int* img, int* res_img,int tmplt_row, int tmplt_col,int img_row,int img_col) {
    __shared__ int s_template[4092];
    __shared__ int s_part[SIZE_STRIPE];
    
    int thx = threadIdx.x;
    int i= blockIdx.x * blockDim.x + threadIdx.x;


    int max_col = SIZE_STRIPE/ tmplt_row;
    int max_row = tmplt_row;
    int index = 0;
    if (thx == 0) {
        for (int j = 0;j < tmplt_row;j++) {
            for (int t = 0;t < tmplt_col;t++) {
                s_template[j * tmplt_col + t] = tmplt[j * tmplt_col + t];
            }
        }

        for (int j = 0;j < max_row;j++) {
            for (int t = 0;t < max_col;t++) {
                index = j * img_col + blockIdx.x * blockDim.x + t;
                    s_part[j * max_col + t] = img[index];
            }
        }
    }
        __syncthreads();

        int c = 0;
        int p = 0;

        for (int j = 0;j < tmplt_row;j++) {
            for (int t = 0;t < tmplt_col;t++) {
                index = j * max_col + t + thx;
                p = s_part[index]; 
                c += s_template[j * tmplt_col + t] * p;
            }
        }

        res_img[i] = c;
}





int main()
{  
    MyImg myimg = MyImg();
    myimg.LoadImg();
    MyImg mytmplt = MyImg();
    mytmplt.LoadImg();
    int* img_thresholding = (int*)malloc(sizeof(int));
    int* tmplt_thresholding = (int*)malloc(sizeof(int));
    int img_rows = myimg.getRows();
    int img_cols = myimg.getCols();
    int img_size = img_rows * img_cols;
    int tmplt_rows = mytmplt.getRows();
    int tmplt_cols = mytmplt.getCols();
    int tmplt_size = tmplt_rows * tmplt_cols;
    int* array_img_bin = (int*)malloc(img_size * sizeof(int));
    int* array_tmplt_bin= (int*)malloc(tmplt_size * sizeof(int));

    if (getThresholding(img_rows, img_cols, myimg.histogram, myimg.array_img, img_thresholding) > 0) {
        return 0;
    }

    if (getThresholding(tmplt_rows, tmplt_cols, mytmplt.histogram, mytmplt.array_img, tmplt_thresholding) > 0) {
        return 0;
    }
       


    getBinary(img_size, myimg.array_img, array_img_bin,img_thresholding,0);

    getBinary(tmplt_size, mytmplt.array_img, array_tmplt_bin, tmplt_thresholding,-1);

    int n1 = 0;
    for (int i = 0;i < tmplt_size;i++) {
        if (array_tmplt_bin[i] == 1) {
            n1++;
        }
    }
    std::cout << "N1: " << n1 << std::endl;


    //printBinImgByArray(tmplt_rows, tmplt_cols, array_tmplt_bin);


    int col = (floor(img_cols / tmplt_cols))* tmplt_cols;
    int row = (floor(img_rows / tmplt_rows))* tmplt_rows;

    std::cout << "Img: r: " << img_rows << " c: " << img_cols<<std::endl;
    std::cout << "Template: r: " << tmplt_rows << " c: " << tmplt_cols << std::endl;
    std::cout << "ROW: " << row << " COL: " << col;

    

    dim3 block((SIZE_STRIPE +1)/ tmplt_rows- tmplt_cols+1);

    dim3 grid(((img_rows * img_cols) + block.x - 1) / block.x);
    cudaError_t cudaStatus;
   
    int res = 0;
    int* dev_array_img_bin;
    int* dev_array_tmplt_bin;

    cudaStatus = cudaMalloc((void**)&dev_array_img_bin, img_size * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev img bin");
    cudaStatus = cudaMemcpy(dev_array_img_bin, array_img_bin, img_size * sizeof(int), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia img bin da host a device");
    cudaStatus = cudaMalloc((void**)&dev_array_tmplt_bin, tmplt_size * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev tmplt bin");
    cudaStatus = cudaMemcpy(dev_array_tmplt_bin, array_tmplt_bin, tmplt_size * sizeof(int), cudaMemcpyHostToDevice);
    res += resultCudaAlloc(cudaStatus, "Errore copia tmplt bin da host a device");

    std::cout << "block:" << ((img_rows * img_cols) + block.x - 1) / block.x << std::endl;


    int* dev_res_img;
    int size_res = img_rows*img_cols;
    int* res_img=(int*)malloc(size_res *sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_res_img, size_res * sizeof(int));
    res += resultCudaAlloc(cudaStatus, "Errore allocazione dev img res");

    correlation << <grid, block >> > (dev_array_tmplt_bin, dev_array_img_bin, dev_res_img, tmplt_rows, tmplt_cols,img_rows,img_cols);

    cudaStatus = cudaMemcpy(res_img, dev_res_img, size_res * sizeof(int), cudaMemcpyDeviceToHost);
    res += resultCudaAlloc(cudaStatus, "-----Errore copia res img da device a host");

    /*
    thrust::host_vector<int> h_vec(size_res);
    thrust::device_vector<int> d_vec(size_res);
    thrust::copy(res_img, res_img + size_res, d_vec.begin());

    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    */




/*
    cv::Mat src_tm = cv::Mat(tmplt_rows, tmplt_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < tmplt_rows; ++i) {
        for (int j = 0; j < tmplt_cols; ++j) {
            // std::cout<<(int)src.at<uchar>(i, j)<<std::endl;
            if (array_tmplt_bin[tmplt_cols * i + j] == -1) {
                src_tm.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }


        }
    }
    cv::imshow("image", src_tm);

    cv::waitKey(0);
*/

    
    cv::Mat src_t = cv::Mat(img_rows, img_cols,CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < img_rows; ++i) {
            for (int j = 0; j < img_cols; ++j) {
                if (array_img_bin[img_cols*i+j] == 0) {
                    src_t.at<cv::Vec3b>(i, j) = cv::Vec3b(255,255,255);
                }
            

            }
        }

     float val;
     int r;
     int c;

     for (int i = 0;i < size_res;i++) {
         val = res_img[i];
        
         r = floor(i / img_cols);
         c = i - r * img_cols;

         if (val>= n1*0.8) {

             cv::rectangle(src_t, cv::Rect(c, r, tmplt_cols, tmplt_rows), cv::Scalar(0, 0, 255));
             cv::putText(src_t, std::to_string((int)((float)val / n1 * 100)), cv::Point(c, r - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0,255), 2);
         }
        else if(val >= n1 * 0.7 && val < n1 * 0.8){
           
             cv::rectangle(src_t, cv::Rect(c, r, tmplt_cols, tmplt_rows), cv::Scalar(0, 255, 0));
             cv::putText(src_t, std::to_string((int)((float)val / n1 * 100)), cv::Point(c, r - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
         }
         else if (val >= n1 * 0.6 && val < n1 * 0.7) {
             cv::rectangle(src_t, cv::Rect(c, r, tmplt_cols, tmplt_rows), cv::Scalar( 255,0, 0));
             cv::putText(src_t, std::to_string((int)((float)val / n1 * 100)), cv::Point(c, r - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
         }
         

         
     }
     cv::imwrite("C:/Users/farmamico_fr/Desktop/progettofinale/Final/x64/Release/result.jpg", src_t);

      cv::resize(src_t, src_t, cv::Size(img_cols*0.50, img_rows*0.50), 0, 0, cv::INTER_LINEAR);
     cv::imshow("image", src_t);

     cv::waitKey(0);
     
     

    
    return 0;
}

