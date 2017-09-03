#pragma once

#include <iostream>
#include <vector>
#include "color.h"

class Image {
protected:
	unsigned w, h;
	std::vector<tuple3> image;

public:
	void init(unsigned w, unsigned h);
	bool load(const char *fn);
	bool save(const char *fn) const;
	inline tuple3 get(int x, int y) const { return image[x + y * w]; }
	inline void set(int x, int y, tuple3 color) { image[x + y * w] = color; }
	inline unsigned getW() const { return w; }
	inline unsigned getH() const { return h; }
};

class ImageLab : public Image {
public:
	bool load(const char *fn);
    bool save(const char *fn) const;
};