
#include "fantasia.h"



class MyImg {
public:
	MyImg(std::string name_file);
	MyImg();
	std::string path = getExePath() + "/img";
	std::string namefile;
	int* histogram;
	int* array_img;
	int getRows();
	int getCols();
	bool LoadImg();

private:
	int rows;
	int cols;
	cv::Mat src;
	void setHistogram();
	

};

