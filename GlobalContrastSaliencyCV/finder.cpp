#include "finder.h"
#include "cropper.h"
#include <algorithm>

void Finder::find() {
	const float overlap = 0.95;
	const float fovStep = 2.0 * CV_PI * (1.0 - overlap);

	float theta;
	float phi;

	Cropper crop;
	std::vector<Frame> frames;

	const float fovModel[] = { 
		104.0 / 180.0 * CV_PI, 
		84.0 / 180.0 * CV_PI, 
		63.0 / 180.0 * CV_PI, 
		47.0 / 180.0 * CV_PI, 
		28.0 / 180.0 * CV_PI 
	};
	const int fovLevel = sizeof(fovModel) / sizeof(float);
	const float aspect = kernelSize.width / kernelSize.height;

	phi = 0.0;
	// 1/3 on the horizon
	for (theta = 0.0; theta < CV_PI * 2.0; theta += fovStep / cos(phi)) {
		for (int i = 0; i < fovLevel; i++) {
			float fov = fovModel[i];
			Frame f1 = Frame(theta, phi + fov / aspect / 6.0, fov, 0.0);
			f1.vertlock = true;
			eval(f1);
			frames.push_back(f1);
			Frame f2 = Frame(theta, phi - fov / aspect / 6.0, fov, 0.0);
			f2.vertlock = true;
			eval(f2);
			frames.push_back(f2);
		}
	}

	// Over the horizon
	phi = CV_PI / 5.5;
	for (theta = 0.0; theta < CV_PI * 2.0; theta += fovStep / cos(phi)) {
		for (int i = 0; i < fovLevel; i++) {
			float fov = fovModel[i];
			Frame f1 = Frame(theta, phi, fov, 0.0);
			eval(f1);
			frames.push_back(f1);
			Frame f2 = Frame(theta, -phi, fov, 0.0);
			eval(f2);
			frames.push_back(f2);
		}
	}

	// Non-maximum suppression
	nonMaxSuppression(frames);

	// Fine-grained search
	for (int i = 0; i < 3; i++) {
		Frame& f = frames[i];
		f = findFrom(f, 10);
	}

	// Non-maximum suppression, again
	// nonMaxSuppression(frames);

	mosts.clear();
	mosts.push_back(frames[0]);
	mosts.push_back(frames[1]);
	mosts.push_back(frames[2]);

	// Normalize frames score
	float sMax = std::min_element(frames.begin(), frames.end())->score;
	float sMin = std::max_element(frames.begin(), frames.end())->score;

	for (int i = 0; i < frames.size(); i++) {
		Frame& f = frames[i];
		f.score = (f.score - sMin) / (sMax - sMin);
	}

	// Draw response map
	for (int i = 0; i < 3; i++) {
		Frame& f = frames[i];

		circle(
			response,
			Point2f(
				f.theta / 2.0 / CV_PI * response.size().width,
				(0.5 - f.phi / CV_PI) * response.size().height
			),
			f.fov / CV_PI * response.size().width * (1.0 - overlap),
			Scalar(0.0, 0.0, f.score * 255.0),
			3
		);
	}
}

void Finder::nonMaxSuppression(std::vector<Frame>& frames) {
	std::sort(frames.begin(), frames.end());
	const float ratio = 1.0;

	for (int i = 0; i < frames.size(); i++) {
		Frame& hold = frames[i];
		Point3f holdV = Point3f(cos(hold.theta), sin(hold.theta), tan(hold.phi));
		holdV /= norm(holdV);

		for (int j = i + 1; j < frames.size(); ) {
			Frame& current = frames[j];
			Point3f curV = Point3f(cos(current.theta), sin(current.theta), tan(current.phi));
			curV /= norm(curV);

			if (norm(curV - holdV) < (current.fov + hold.fov) / 4.0 * ratio) {
				frames.erase(frames.begin() + j);
				continue;
			}

			j++;
		}
	}
}

void Finder::cut(Mat src, Mat& dest, Size outSize, int rank) {
	Cropper crop;
	crop.crop(src, dest, outSize, mosts[rank].theta, mosts[rank].phi, mosts[rank].fov);
}

void Finder::eval(Frame& f) {
	if (type == 0) {
		evalKernelFit(f);
	}
	else if (type == 1) {
		evalContrast(f);
	}
}

void Finder::evalKernelFit(Frame& f) {
	Mat patchSal;
	Cropper crop;
	crop.crop(sal, patchSal, kernelSize, f.theta, f.phi, f.fov);

	float scoreFin = FLT_MIN;

	for (int k = 0; k < kernelSet.size(); k++) {
		float score = 0.0;
		Mat patchSalNew = patchSal.clone();
		patchSalNew = patchSalNew - kernelSet[k].kernels[0];
		patchSalNew = patchSalNew.mul(patchSal);
		patchSalNew = patchSalNew.mul(kernelSet[k].kernels[1]);
		score = 1.0 - sum(patchSalNew)(0) / sum(kernelSet[k].kernels[1])(0);

		if (score > scoreFin)
			scoreFin = score;
	}

	f.score = scoreFin * abs(1.0 - f.phi / CV_PI * 2.0);
}

void Finder::evalContrast(Frame& f) {
	Mat patchLabel;
	Mat patchSal;
	Cropper crop;
	crop.crop(label, patchLabel, kernelSize, f.theta, f.phi, f.fov);
	crop.crop(sal, patchSal, kernelSize, f.theta, f.phi, f.fov);

	float scoreFin = FLT_MIN;

	for (int k = 0; k < kernelSet.size(); k++) {
		float score = 0.0;
		std::vector<Mat>& kernels = kernelSet[k].kernels;

		std::vector<Histogram> histograms;
		for (int i = 0; i < kernels.size(); i++) {
			Histogram hist = gcs->calculateHistogram(patchLabel, kernels[i]);
			histograms.push_back(hist);
		}

		for (int i = 0; i < kernels.size(); i++) {
			for (int j = 0; j < i; j++) {
				score += gcs->calculateHistogramDistance(histograms[i], histograms[j]);
			}
		}

		for (int i = 0; i < kernels.size(); i++) {
			score -= gcs->calculateHistogramDistance(histograms[i], histograms[i]);
		}

		score /= (kernels.size() - 1); // Remove partitioning effect

		Mat patchSalNew;
		patchSal.convertTo(patchSalNew, CV_8U, 255.0);
		patchSalNew = patchSalNew.mul(kernels[0], 1.0 / 255.0);

		score *= (float)sum(patchSalNew)(0) / sum(kernels[0])(0);

		if (score > scoreFin) {
			scoreFin = score;
		}
	}

	f.score = scoreFin * abs(1.0 - f.phi / CV_PI * 2.0);
}

Frame Finder::findFrom(Frame f, int step) {
	const float stride = 1.0 / 180.0 * CV_PI;
	const float stridefov = 1.02;

	while (step) {
		std::vector<Frame> frames;

		Frame fUp = f; fUp.phi += stride;
		Frame fDown = f; fDown.phi -= stride;
		Frame fLeft = f; fLeft.theta -= stride;
		Frame fRight = f; fRight.theta += stride;
		Frame fPush = f; if (fPush.fov < fovMax) fPush.fov *= stridefov;
		Frame fPull = f; if (fPull.fov > fovMin) fPull.fov /= stridefov;

		if (!f.vertlock) {
			frames.push_back(fUp);
			frames.push_back(fDown);
		}
		frames.push_back(fLeft);
		frames.push_back(fRight);
		frames.push_back(fPush);
		frames.push_back(fPull);

		for (int i = 0; i < frames.size(); i++)
			eval(frames[i]);

		// Actually min element is the highest scored frame by definition of operator<
		Frame fNew = *std::min_element(frames.begin(), frames.end());

		if (fNew.score > f.score) {
			f = fNew;
			step--;
		}
		else
			break;
	}

	return f;
}

void Finder::loadKernel(Mat kernel) {
	if (type == 0) {
		loadKernelKernelFit(kernel);
	}
	else if (type == 1) {
		loadKernelContrast(kernel);
	}
}

void Finder::loadKernelKernelFit(Mat kernel) {
	CV_Assert(kernel.type() == CV_8UC3);

	KernelSet ks;

	Mat kernelSplit[3];
	split(kernel, kernelSplit);

	kernelSize = kernel.size();

	kernelSplit[2].convertTo(kernelSplit[2], CV_32F, 1.0 / 255.0);
	kernelSplit[1].convertTo(kernelSplit[1], CV_32F, 1.0 / 255.0);
	ks.kernels.push_back(kernelSplit[2]);
	ks.kernels.push_back(kernelSplit[1]);

	kernelSet.push_back(ks);
}

void Finder::loadKernelContrast(Mat kernel) {
	CV_Assert(kernel.type() == CV_8UC3);

	KernelSet ks;

	kernel = kernel.clone();
	resize(kernel, kernel, kernelSize);

	const Vec3b colors[] = {
		Vec3b(0, 0, 255),
		Vec3b(0, 255, 255),
		Vec3b(0, 255, 0),
		Vec3b(255, 255, 0),
		Vec3b(255, 0, 0),
		Vec3b(255, 0, 255)
	};
	const int colorsLength = sizeof(colors) / sizeof(Vec3b);

	for (int i = 0; i < colorsLength; i++) {
		Mat tmp = kernel.clone();

		bitwise_xor(tmp, colors[i], tmp);
		cvtColor(tmp, tmp, CV_BGR2GRAY);
		threshold(tmp, tmp, 1.0, 255.0, THRESH_BINARY);
		bitwise_not(tmp, tmp);

		if (countNonZero(tmp) == 0)
			break;
		
		ks.kernels.push_back(tmp);
	}

	kernelSet.push_back(ks);
}

void Finder::setSource(Mat sal, Mat label, GCS *gcs) {
	CV_Assert(sal.type() == CV_32F);
	CV_Assert(label.type() == CV_32S);

	this->sal = sal;
	this->label = label;
	this->gcs = gcs;
}

void Finder::setResponseBackground(Mat bg) {
	response = bg.clone();
	resize(response, response, responseSize);
}
