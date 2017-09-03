#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "gcs.h"

using namespace cv;

struct Frame {
	float theta, phi, fov, score;
	bool vertlock = false;
	Frame(float theta, float phi, float fov, float score) : theta(theta), phi(phi), fov(fov), score(score) {}
	bool operator<(const Frame& other) const { return score > other.score; }
};

class Finder {
private:
	std::vector<Frame> mosts;
	Mat response;
	Size responseSize;

	Mat sal, label;
	GCS *gcs;

	float fovMax, fovMin;

	void eval(Frame& f);
	void evalKernelFit(Frame& f);
	void evalContrast(Frame& f);

	void nonMaxSuppression(std::vector<Frame>& frames);

	Size kernelSize;
	struct KernelSet {
		std::vector<Mat> kernels;
	};
	std::vector<KernelSet> kernelSet;

	int type;

public:
	Finder(int type) { 
		responseSize = Size(600, 300); 
		kernelSize = Size(200, 150);
		fovMax = 104.0 / 180.0 * CV_PI;
		fovMin = 28.0 / 180.0 * CV_PI;
		this->type = type;
	}
	void find();
	Frame findFrom(Frame f, int step);
	void cut(Mat src, Mat& dest, Size outSize, int rank);
	void loadKernel(Mat kernel);
	void loadKernelKernelFit(Mat kernel);
	void loadKernelContrast(Mat kernel);
	void setSource(Mat sal, Mat label, GCS *gcs);
	void setResponseBackground(Mat bg);

	void setType(int type) {
		this->type = type;
	}
	Mat getResponse() { return response; }
};