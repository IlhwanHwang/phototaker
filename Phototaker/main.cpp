#include "ImageLab.h"
#include "gcs.h"
#include "cropper.h"
#include "patch.h"
#include "finder.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>

using namespace cv;

static const int FRAME_MAX = 50;
static const int OUTPUT_WIDTH = 320;
static const int OUTPUT_HEIGHT = 160;
static const int FINAL_WIDTH = 640;
static const int FINAL_HEIGHT = 480;
static const float FOV_MULTIPLYER = 6.0f;
static const float FOV_MAX = 104.0f;

int main(int argc, const char **argv) {
	
	if (argc < 2) {
		// std::cout << "No input file" << std::endl;
		// return 0;
	}
	
	// Configurations
	bool enforcePhotoComposition = true;
	bool fovLimit = true;
	bool centerAssumption = true;

	std::string constraintImageFileName = "constraint.png";

	Finder fnd(1);

	for (int i = 2; i < argc; i++) {
		if (strcmp(argv[i], "-c") == 0) {
			enforcePhotoComposition = false;
		}
		if (strcmp(argv[i], "-f") == 0) {
			fovLimit = false;
		}
		if (strcmp(argv[i], "-a") == 0) {
			centerAssumption = false;
		}
		if (strcmp(argv[i], "-k") == 0) {
			Mat kernel = imread(argv[i + 1]);
			fnd.loadKernel(kernel);
			i++;
		}
		if (strcmp(argv[i], "--constraint") == 0) {
			constraintImageFileName = argv[i + 1];
			i++;
		}
		if (strcmp(argv[i], "--kernelfit") == 0) {
			fnd.setType(0);
		}
		if (strcmp(argv[i], "--contrast") == 0) {
			fnd.setType(1);
		}
	}

	if (argc < 2) {
		Mat kernel;
		kernel = imread("kernel_contrast.png");
		fnd.loadKernel(kernel);
		kernel = imread("kernel_contrast_inv.png");
		fnd.loadKernel(kernel);
	}

	{
		Mat frameCapture;
		
		std::string raw;
		if (argc < 2) {
			raw = "images/good_city5.jpg";
		}
		else {
			raw = argv[1];
		}
		std::string fname = raw.substr(0, raw.find_last_of("."));
		std::string ext = raw.substr(raw.find_last_of("."));

		std::string input = fname + ext;
		std::string outputResponse = "output/" + fname + "_response" + ext;
		std::string outputSaliency = "output/" + fname + "_saliency" + ext;
		std::string outputFinal1 = "output/" + fname + "_final_1" + ext;
		std::string outputFinal2 = "output/" + fname + "_final_2" + ext;
		std::string outputFinal3 = "output/" + fname + "_final_3" + ext;
		
		frameCapture = imread(input);
		//cap >> frameCapture;
		//imshow(windowName, frameCapture);

		GCS gcs;
		Mat imgIn;
		Mat imgLabel;
		Mat imgIn2;
		Mat imgOut;
		Mat imgPDS;

		resize(frameCapture, imgIn, Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

		imgOut = Mat::zeros(imgIn.size().height, imgIn.size().width, CV_8UC3);
		imgIn2 = Mat::zeros(imgIn.size().height, imgIn.size().width, CV_32FC3);

		gcs.setReferenceSize(fmax((float)imgIn.size().width, (float)imgIn.size().height));

		// Set FOV limit
		if (fovLimit)
			gcs.setLimitSize(FOV_MAX * FOV_MAX / FOV_MULTIPLYER / FOV_MULTIPLYER * OUTPUT_WIDTH * OUTPUT_WIDTH / 360.0f / 360.0f);
			//gcs.setLimitSize(FOV_MAX * FOV_MAX / FOV_MULTIPLYER / FOV_MULTIPLYER * OUTPUT_WIDTH * OUTPUT_HEIGHT);
		else
			gcs.setLimitSize(FLT_MAX);

		// Set center assumption
		gcs.setCenterAssumption(centerAssumption);

		std::cout << "Segmenting..." << std::endl;
		gcs.buildSegmentation(imgIn);

		std::cout << "Posterizing..." << std::endl;
		gcs.posterize(imgIn, imgLabel, 40);

		std::cout << "Building histogram..." << std::endl;
		gcs.buildHistogram(imgLabel);

		std::cout << "Calculating saliency..." << std::endl;
		gcs.calculateSaliency();
		Mat imgGCS = gcs.getSaliency();
		Mat imgScale = gcs.getScaleFactor();

		PDS pds;
		std::cout << "Patch saliency..." << std::endl;
		pds.calculate(frameCapture, Size(OUTPUT_WIDTH, OUTPUT_HEIGHT), imgPDS);
		// imgPDS = Mat::ones(imgGCS.size(), CV_32F);

		Mat imgSal;
		multiply(imgGCS, imgPDS, imgSal);
		multiply(imgScale, imgSal, imgSal);

		Mat img;
		imgGCS.convertTo(img, CV_8U, 255.0);
		imwrite("GCS.png", img);
		imgPDS.convertTo(img, CV_8U, 255.0);
		imwrite("PDS.png", img);
		imgSal.convertTo(img, CV_8U, 255.0);
		imwrite("Sal.png", img);

		Mat constraint = imread(constraintImageFileName);
		resize(constraint, constraint, imgSal.size());
		cvtColor(constraint, constraint, CV_BGR2GRAY);
		constraint.convertTo(constraint, CV_32F, 1.0 / 255.0);
		// imgSal = imgSal.mul(constraint);
		normalize(imgSal, imgSal, 1.0, 0.0, NORM_MINMAX);
		threshold(imgSal, imgSal, 0.3, 1.0, THRESH_BINARY);

		imgSal.convertTo(img, CV_8U, 255.0);
		imwrite("thr.png", img);

		fnd.setSource(imgSal, imgLabel, &gcs);
		fnd.setResponseBackground(frameCapture);

		std::cout << "Fitting..." << std::endl;
		fnd.find();

		imgSal.convertTo(imgSal, CV_8UC1, 255.0);

		imwrite(outputSaliency, imgSal);
		imwrite(outputResponse, fnd.getResponse());

		fnd.cut(frameCapture, imgOut, Size(FINAL_WIDTH, FINAL_HEIGHT), 0);
		imwrite(outputFinal1, imgOut);

		fnd.cut(frameCapture, imgOut, Size(FINAL_WIDTH, FINAL_HEIGHT), 1);
		imwrite(outputFinal2, imgOut);

		fnd.cut(frameCapture, imgOut, Size(FINAL_WIDTH, FINAL_HEIGHT), 2);
		imwrite(outputFinal3, imgOut);

		/*

		Mat imgOut2;

		Point3f vecFront;
		vecFront = gcs.getMaxResponsePosition();

		const float pi = 3.14159265;
		float fov, fovv, fovRad, fovvRad;

		float theta;
		if (vecFront.x == 0.0 && vecFront.y > 0.0)
			theta = pi / 2.0;
		else if (vecFront.x == 0.0 && vecFront.y < 0.0)
			theta = -pi / 2.0;
		else
			theta = atan(vecFront.y / vecFront.x) + (vecFront.x > 0 ? 0.0 : pi);

		float phi;
		phi = asin(vecFront.z);

		if (gcs.getWeightSTDev() > 550.0f) {
			fov = 90.0f;
			fovv = fov * ((float)FINAL_HEIGHT / FINAL_WIDTH);
			fovRad = fov * pi / 180.0f;
			fovvRad = fovv * pi / 180.0f;

			phi = fovvRad / 6.0;

			vecFront = Point3f(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi));
		}
		else {
			fov = FOV_MULTIPLYER * 360.0f * sqrt(gcs.getMaxResponseArea()) / OUTPUT_WIDTH;
			fov *= 100.0f / gcs.getSalSTDev();

			fovv = fov * ((float)FINAL_HEIGHT / FINAL_WIDTH);
			fovRad = fov * pi / 180.0f;
			fovvRad = fovv * pi / 180.0f;

			if (enforcePhotoComposition)
			{ // Try to enforce rules of photographs
				bool thirdsRules = false;

				// 1. Strict align on horizon within 9 degree
				if (abs(phi) < pi * 0.05) {
					phi = 0.0;
					thirdsRules = true;
				}

				// 2. 1/3 align on horizon
				if (phi < fovvRad / 2.0 && phi > pi * 0.05) {
					phi = fovvRad / 6.0;
					thirdsRules = true;
				}
				if (phi > -fovvRad / 2.0 && phi < -pi * 0.05) {
					phi = -fovvRad / 6.0;
					thirdsRules = true;
				}

				// 3. If horizon is aligned, so is object.
				if (thirdsRules) {
					theta += fovRad / 6.0f;
				}

				// 4. If object superior, place it over.
				if (phi > fovvRad / 2.0f) {
					phi -= fovvRad / 6.0f;
				}

				// 5. If object inferir, place it under.
				if (phi < -fovvRad / 2.0f) {
					phi += fovvRad / 6.0f;
				}

				vecFront = Point3f(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi));
			}
		}
		
		Cropper crp;
		crp.crop(frameCapture, imgOut2, Size(FINAL_WIDTH, FINAL_HEIGHT), vecFront, fov);

		//imshow(windowName, imgOut2);
		imwrite(outputFinal, imgOut2);
		//waitKey(0);

		//wrt << imgOut;

		*/
	}

	return 0;
}