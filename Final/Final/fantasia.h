

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <device_functions.h>

#include <cstdio>
#include <cstdlib>




#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <string>



const int MAX_INT = 256;
const int SIZE_STRIPE = 8196;

std::string getExePath();
int resultCudaAlloc(cudaError_t cudaStatus, std::string msg);

void confrontoW(int* histogram, float *sigma);

void printBinImgByArray(int rows, int cols, int* array_bin);



