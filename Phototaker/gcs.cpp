#include "ImageLab.h"
#include "gcs.h"
#include <cmath>
#include <map>
#include <algorithm>
#include <cfloat>
#include "segmentation.h"

void GCS::posterize(Mat src, Mat& dest, int k) {
	CV_Assert(src.type() == CV_8UC3);

	colorBinSize = k;

	Mat newSrc;
	src.convertTo(newSrc, CV_32FC3, 1.0 / 255.0);
	cvtColor(newSrc, newSrc, CV_BGR2Luv);

	// Vectorize colors
	Mat colorPoints = Mat::zeros(src.size().width * src.size().height, 3, CV_32FC1);

	for (int y = 0, i = 0; y < src.size().height; y++) {
		for (int x = 0; x < src.size().width; x++, i++) {
			for (int c = 0; c < 3; c++) {
				colorPoints.at<float>(i, c) = src.at<Vec3b>(y, x)[c];
			}
		}
	}

	Mat labels, centers;
	kmeans(colorPoints, k, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	// Vector list to color list
	std::vector<Vec3f> colorCenters;
	for (int i = 0; i < centers.size().height; i++) {
		Vec3f col;
		for (int c = 0; c < 3; c++) {
			col(c) = centers.at<float>(i, c);
		}
		colorCenters.push_back(col);
	}

	dest = Mat::zeros(src.size(), CV_32S);

	// Save color numbers
	for (int y = 0, i = 0; y < src.size().height; y++) {
		for (int x = 0; x < src.size().width; x++, i++) {
			int label = labels.at<int>(i);
			dest.at<int>(y, x) = label;
		}
	}

	// Convert color space ?
	// cvtColor(dest, dest, CV_Luv2BGR);
	// dest.convertTo(dest, CV_8UC3, 255.0);

	// Build difference map
	// k == colorCenters.size()
	differenceMap = Mat::zeros(Size(k, k), CV_32F);
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			float dist;
			dist = norm(colorCenters[i] - colorCenters[j]);
			differenceMap.at<float>(i, j) = dist;
		}
	}
}

void GCS::buildHistogram(Mat image) {
	CV_Assert(image.type() == CV_32S); // Assume label number matrix

	std::set<SegComponent*, lessComp>& cs = segSegment.getComponents();
	std::set<SegComponent*, lessComp>::iterator itor;

	// std::map<tuple3, float> histomapTotal;

	int id = 0;

	histogramTotal.init(colorBinSize);

	for (itor = cs.begin(); itor != cs.end(); itor++) {
		segment.push_back(GCSRegion(id++, colorBinSize));
		GCSRegion& gcsr = *(segment.end() - 1);
		SegComponent& segc = **itor;

		// Accumulating histogram elements
		// std::map<tuple3, float> histomap;

		gcsr.w = 0.0;

		for (int i = 0; i < segc.vertices.size(); i++) {
			SegVertex& v = *segc.vertices[i];
			gcsr.vertices.push_back(position(v.x, v.y));

			gcsr.pos += Point3f(cos(v.theta) * cos(v.phi), sin(v.theta) * cos(v.phi), sin(v.phi)) * v.w;
			gcsr.w += v.w;

			int label = image.at<int>(v.y, v.x);
			gcsr.histogram.gram[label] += v.w;
			histogramTotal.gram[label] += v.w;

			/*
			Vec3b col = image.at<Vec3f>(v.y, v.x);
			tuple3 lab = tuple3(col[0], col[1], col[2]);

			if (histomap.find(lab) == histomap.end())
				histomap[lab] = v.w;
			else
				histomap[lab] += v.w;

			if (histomapTotal.find(lab) == histomapTotal.end())
				histomapTotal[lab] = v.w;
			else
				histomapTotal[lab] += v.w;
			*/
		}

		gcsr.pos /= norm(gcsr.pos);

		/*
		// histogram elements to vector histogram
		std::map<tuple3, float>::iterator mitor;
		
		for (mitor = histomap.begin(); mitor != histomap.end(); mitor++)
			gcsr.histogram.push_back(histoElem(mitor->first, mitor->second));
			*/
	}

	// histogram elements to vector histogram (total histogram)
	/*
	std::map<tuple3, float>::iterator mitor;

	for (mitor = histomapTotal.begin(); mitor != histomapTotal.end(); mitor++)
		histogramTotal.push_back(histoElem(mitor->first, mitor->second));
		*/
}

void GCS::buildSegmentation(Mat image) {
	segSegment.init(image, sqrt(image.size().width * image.size().height), 0.001f);
	segSegment.build();
}

void GCS::outputSegmentation(Mat image) {
	segSegment.output(image);
}

/*
float GCS::regionDifference(const GCSRegion& r1, const GCSRegion& r2) const {
	float diff = 0.0;

	for (int i = 0; i < r1.histogram.size(); i++) {
		const histoElem& h1 = r1.histogram[i];
		for (int j = 0; j < r2.histogram.size(); j++) {
			const histoElem& h2 = r2.histogram[j];
			diff += labdist(h1.first, h2.first) * (float)h1.second * (float)h2.second;
		}
	}

	return diff;
}
*/

/*
void GCS::calculateRegionDifference() {
	for (int i = 0; i < segment.size(); i++) {
		for (int j = i; j < segment.size(); j++) {
			if (i == j) {
				regionDef[position(i, j)] = 0.0f;
			}
			else {
				regionDef[position(i, j)] = regionDifference(segment[i], segment[j]);
			}
		}
	}
	
	for (int i = 0; i < segment.size(); i++) {
		for (int j = 0; j < i; j++) {
			regionDef[position(i, j)] = regionDef[position(j, i)];
		}
	}
}
*/

void GCS::normalizeSaliency() {
	float sMax = FLT_MIN;
	float sMin = FLT_MAX;

	// Detect maximum and minumum saliency
	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		if (r.saliency > sMax) {
			sMax = r.saliency;
			segMax = &r;
		}
		if (r.saliency < sMin) {
			sMin = r.saliency;
		}
	}

	// Normalize
	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		r.saliency = (r.saliency - sMin) / (sMax - sMin);
	}
}

void GCS::calculateSaliency() {
	// Calculate weight variance
	weightSTDev = 0.0f;
	float weightMean = 0.0f;

	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		weightMean += r.w;
	}
	weightMean /= (float)segment.size();

	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		const float dev = r.w - weightMean;
		weightSTDev += dev * dev;
	}
	weightSTDev /= (float)segment.size();
	weightSTDev = sqrt(weightSTDev);

	printf("STDEV WEIGHT: %f\n", weightSTDev);

	
	float sceneSize = 0.0f;

	salSTDev = 0.0f;
	float salMean = 0.0f;

	// Calculate saliency from global histogram
	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		r.saliency = 0.0f;

		for (int i = 0; i < colorBinSize; i++) {
			const float h = r.histogram.gram[i];
			for (int j = 0; j < colorBinSize; j++) {
				const float ht = histogramTotal.gram[j] - r.histogram.gram[j];
				r.saliency += differenceMap.at<float>(i, j) * h * ht;
			}
		}

		r.saliency /= r.w;
		sceneSize += r.w;
		salMean += r.saliency;
	}

	salMean /= sceneSize;

	// Calculate variation of saliency before normalization
	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		const float dev = r.saliency / r.w - salMean;
		salSTDev += dev * dev * r.w;
	}

	salSTDev /= sceneSize;
	salSTDev = sqrt(salSTDev);
	printf("STDEV SAL: %f\n", salSTDev);
}

Mat GCS::getSaliency() {
	Mat out = Mat::zeros(segSegment.getSize(), CV_32FC1);

	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		for (int j = 0; j < r.vertices.size(); j++) {
			position& v = r.vertices[j];
			out.at<float>(v.second, v.first) = r.saliency;
		}
	}

	normalize(out, out, 0.0, 1.0, cv::NORM_MINMAX);

	return out;
}

Mat GCS::getScaleFactor() {
	Mat out = Mat::zeros(segSegment.getSize(), CV_32FC1);

	for (int i = 0; i < segment.size(); i++) {
		GCSRegion& r = segment[i];
		for (int j = 0; j < r.vertices.size(); j++) {
			position& v = r.vertices[j];
			out.at<float>(v.second, v.first) = -r.w;
		}
	}

	normalize(out, out, 0.0, 1.0, cv::NORM_MINMAX);

	return out;
}

Histogram GCS::calculateHistogram(Mat label, Mat mask) {
	int histSize[] = { colorBinSize };
	float hranges[] = { 0, colorBinSize };
	const float* ranges[] = { hranges };
	Mat hist;
	
	int channels[] = { 0 };

	Mat newLabel = label.clone();
	newLabel.convertTo(newLabel, CV_32F);
	calcHist(&newLabel, 1, channels, mask, hist, 1, histSize, ranges, true, false);

	Histogram out;
	out.init(colorBinSize);
	for (int i = 0; i < colorBinSize; i++) {
		out.gram[i] = hist.at<float>(i);
	}

	return out;
}

float GCS::calculateHistogramDistance(Histogram h1, Histogram h2) {
	float dist = 0.0;

	for (int i = 0; i < colorBinSize; i++) {
		const float a = h1.gram[i];
		for (int j = 0; j < colorBinSize; j++) {
			const float b = h2.gram[j];
			dist += differenceMap.at<float>(i, j) * a * b;
		}
	}

	return dist;
}