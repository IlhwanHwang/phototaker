#pragma once

#include <opencv2/opencv.hpp>

class PDS {
private:
	size_t patchSize;
	float patchFOV;
	int level;
public:
	void calculate(cv::Mat src, cv::Size outSize, cv::Mat& dest);
};