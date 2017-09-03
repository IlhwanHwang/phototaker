#pragma once

#include <iostream>
#include <vector>
#include <map>
#include "segmentation.h"
#include <opencv2/opencv.hpp>

using namespace cv;

typedef std::pair<int, int> position;

struct Histogram {
	std::vector<float> gram;

	void init(int colorBinSize) {
		gram.clear();
		for (int i = 0; i < colorBinSize; i++) {
			gram.push_back(0.0);
		}
	}
};

struct GCSRegion {
	int id;
	std::vector<position> vertices;
	Histogram histogram;
	float saliency;
	float w;
	Point3f pos;
	GCSRegion(int id, int colorBinSize) : id(id), saliency(0.0f), pos(Point3f(0.0f, 0.0f, 0.0f)) {
		histogram.init(colorBinSize);
	}
};

class GCS {
	Segmentation segSegment;
	GCSRegion *segMax;
	std::vector<GCSRegion> segment;
	std::map<position, float> regionDef;
	Histogram histogramTotal;
	float refSize, limitSize;
	bool centerAssumption;
	float salSTDev, weightSTDev;
	Mat differenceMap;
	int colorBinSize;

	float regionDifference(const GCSRegion& r1, const GCSRegion& r2) const;

public:
	void setReferenceSize(float size) { refSize = size; };
	void setLimitSize(float size) { limitSize = size; }
	void setCenterAssumption(bool center) { centerAssumption = center; }
	void posterize(Mat src, Mat& dest, int k);
	void buildHistogram(Mat image);
	void buildSegmentation(Mat image);
	void outputSegmentation(Mat image);
	void calculateRegionDifference();
	void calculateSaliency();
	void normalizeSaliency();
	Histogram calculateHistogram(Mat label, Mat mask);
	float calculateHistogramDistance(Histogram h1, Histogram h2);

	Point3f getMaxResponsePosition() {
		return segMax->pos;
	}
	float getMaxResponseArea() {
		return segMax->w;
	}
	float getSalSTDev() { return salSTDev; }
	float getWeightSTDev() { return weightSTDev; }
	Mat getSaliency();
	Mat getScaleFactor();
};