#include "patch.h"
#include "cropper.h"

void PDS::calculate(cv::Mat src, cv::Size outSize, cv::Mat& dest) {
	CV_Assert(src.depth() == CV_8U);

	if (src.channels() == 3) {
		cv::cvtColor(src, src, cv::COLOR_RGB2GRAY);
	}

	patchSize = 12;
	patchFOV = 5.0 / 180 * CV_PI;
	level = 3;

	cv::Mat patches;
	cv::Size patchDimension = cv::Size(patchSize, patchSize);

	const float overlap = 0.5;
	const float fovStep = patchFOV * (1.0 - overlap);

	int row = 0;
	float theta = 0.0;
	float phi = -floor(CV_PI / 2.0 / fovStep) * fovStep;

	// Calculate total rows
	for (; phi < CV_PI / 2.0; phi += fovStep) {
		for (theta = 0.0; theta < CV_PI * 2.0; theta += fovStep / cos(phi)) {
			row++;
		}
	}

	patches = cv::Mat::zeros(row * level, patchSize * patchSize, CV_32FC1);

	Cropper crop;

	// Sample patches
	row = 0;
	float curPatchFov = patchFOV;
	for (int i = 0; i < level; i++) {
		phi = -floor(CV_PI / 2.0 / fovStep) * fovStep;
		for (; phi < CV_PI / 2.0; phi += fovStep) {
			for (theta = 0.0; theta < CV_PI * 2.0; theta += fovStep / cos(phi)) {
				cv::Mat patch;
				crop.crop(src, patch, patchDimension, theta, phi, curPatchFov);
				patch = patch.reshape(1, 1);
				patch.row(0).convertTo(patches.row(row), CV_32FC1, 1.0 / 255.0);
				row++;
			}
		}
		curPatchFov *= 2.0;
	}

	// Solve PCA
	cv::PCA pca(patches, cv::noArray(), cv::PCA::DATA_AS_ROW, 0.25);

	// Sample patches and get distinctness
	dest = cv::Mat::zeros(outSize, CV_32FC1);

	for (int y = 0; y < outSize.height; y++) {
		const float phi = -(((float)y + 0.5) / outSize.height * 2.0 - 1.0) * CV_PI / 2.0;
		const float coverage = 2.0 * CV_PI / cos(phi) / outSize.width;
		for (float theta = 0.0; theta < CV_PI * 2.0; theta += coverage) {
			float s = 0.0;
			curPatchFov = patchFOV;
			for (int i = 0; i < level; i++) {
				cv::Mat patch;
				crop.crop(src, patch, patchDimension, theta, phi, curPatchFov);
				patch = patch.reshape(1, 1);
				patch.convertTo(patch, CV_32FC1);
				cv::Mat project = pca.backProject(pca.project(patch));

				s += cv::norm(patch - project);
				curPatchFov *= 2.0;
			}
			const float x1 = (theta - coverage / 2.0) / CV_PI / 2.0 * outSize.width;
			const float x2 = (theta + coverage / 2.0) / CV_PI / 2.0 * outSize.width;
			cv::line(dest, cv::Point2d(x1, y), cv::Point2d(x2, y), s);
			cv::line(dest, cv::Point2d(x1 + outSize.width, y), cv::Point2d(x2 + outSize.width, y), s);
		}
	}

	cv::normalize(dest, dest, 0.0, 1.0, cv::NORM_MINMAX);
}