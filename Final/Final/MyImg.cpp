#include "MyImg.h"










/********Class my Img**********/
MyImg::MyImg(std::string name_file) {
    namefile=name_file;
}

MyImg::MyImg() {
    //ho avuto problemi con la npp thrust, per ora faccio a mano opencv
    std::cout << "Inserire il nome dell'immagine da analizzare (il file deve essere posizionato in: " << path << ")" << std::endl;
    std::cin >> namefile;
}


int MyImg::getRows() {
    return src.rows;
}
int MyImg::getCols() {
    return src.cols;
}

bool MyImg::LoadImg() {
    src = cv::imread(path + "/" + namefile, cv::IMREAD_GRAYSCALE); // Load an image
    if (src.empty())
    {
        std::cout << "Cannot read the image: " << namefile << std::endl;
        return false;
    }
    else {
        rows = this->getRows();
        cols = this->getCols();
        array_img = (int*)malloc(rows * cols * sizeof(int));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                array_img[i * cols + j] = (int)src.at<uchar>(i, j);
            }
        }
        if (!array_img)
        {
            std::cout << "Errore - Memoria allocata per l'img non disponibile";
            return false;
        }
        else {
            setHistogram();
            return true;
        }
    }
   /* int t = cv::threshold(src, src, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << t << "OPENCV" << std::endl;
    cv::imshow("image", src);
    cv::waitKey(0);*/
}





void MyImg::setHistogram() {
    histogram = (int*)malloc(MAX_INT * sizeof(int));
    for (int i = 0;i < MAX_INT;i++) {
        histogram[i] = 0;
    }
}

