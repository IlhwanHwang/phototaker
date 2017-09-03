#include "ImageLab.h"

void Image::init(unsigned w, unsigned h) {
	image.clear();

	this->w = w;
	this->h = h;

	for (unsigned j = 0; j < h; j++) {
		for (unsigned i = 0; i < w; i++) {
			image.push_back(tuple3(0.0f));
		}
	}
}

bool Image::load(const char *fn) {
	std::vector<unsigned char> png;
	std::vector<unsigned char> cimage;
	unsigned error;

	error = lodepng::load_file(png, fn);
	if (error) {
		std::cout << "Error " << error << " loading image " << fn << std::endl;
		return false;
	}

	error = lodepng::decode(cimage, w, h, png);
	if (error) {
		std::cout << "Error " << error << " loading image " << fn << std::endl;
		return false;
	}

	image.clear();
	for (int i = 0; i < cimage.size(); i += 4) {
		image.push_back(tuple3(cimage[i], cimage[i + 1], cimage[i + 2]));
	}

	std::cout << "Image: " << fn << " is successfully loaded" << std::endl;

	return true;
}

bool Image::save(const char *fn) const {
    std::vector<unsigned char> cimage;
    std::vector<unsigned char> png;
    unsigned error;
    
    for (int i = 0; i < image.size(); i++) {
		tuple3 rgb = image[i];
        
        cimage.push_back((unsigned char)(rgb.x * 255.0));
        cimage.push_back((unsigned char)(rgb.y * 255.0));
        cimage.push_back((unsigned char)(rgb.z * 255.0));
        cimage.push_back(255);
    }
    error = lodepng::encode(png, cimage, w, h);
    if (error) {
        std::cout << "Error " << error << " saving image " << fn << std::endl;
        return false;
    }
    lodepng::save_file(png, fn);

	return true;
}

bool ImageLab::load(const char *fn) {
	bool pass;
	pass = Image::load(fn);

	if (!pass)
		return false;

	for (int i = 0; i < image.size(); i++) {
		image[i] = rgb2lab(image[i]);
	}

	return true;
}

static inline float clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

bool ImageLab::save(const char *fn) const {
	std::vector<unsigned char> cimage;
	std::vector<unsigned char> png;
	unsigned error;

	for (int i = 0; i < image.size(); i++) {
		tuple3 lab = image[i];
		tuple3 rgb = lab2rgb(lab);

		rgb.x = clamp(rgb.x, 0.0f, 1.0f);
		rgb.y = clamp(rgb.y, 0.0f, 1.0f);
		rgb.z = clamp(rgb.z, 0.0f, 1.0f);

		cimage.push_back((unsigned char)(rgb.x * 255.0));
		cimage.push_back((unsigned char)(rgb.y * 255.0));
		cimage.push_back((unsigned char)(rgb.z * 255.0));
		cimage.push_back(255);
	}
	error = lodepng::encode(png, cimage, w, h);
	if (error) {
		std::cout << "Error " << error << " saving image " << fn << std::endl;
		return false;
	}
	lodepng::save_file(png, fn);

	return true;
}

