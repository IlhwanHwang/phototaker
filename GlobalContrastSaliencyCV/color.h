#pragma once

struct tuple3 {
	float x, y, z;
	tuple3(float x, float  y, float z) : x(x), y(y), z(z) {}
	tuple3(float i) : x(i), y(i), z(i) {}
	tuple3(unsigned char cx, unsigned char cy, unsigned char cz) : x(cx / 255.0), y(cy / 255.0), z(cz / 255.0) {}
	bool operator< (const tuple3& other) const {
		const tuple3& a = *this;
		const tuple3& b = other;

		if (a.x < b.x)
			return true;
		if (a.x > b.x)
			return false;

		if (a.y < b.y)
			return true;
		if (a.y > b.y)
			return false;

		if (a.z < b.z)
			return true;
		if (a.z > b.z)
			return false;

		return false;
	}
	bool operator> (const tuple3& other) const {
		return !(*this < other);
	}
};

tuple3 rgb2xyz(tuple3 rgb);
tuple3 xyz2lab(tuple3 xyz);
inline tuple3 rgb2lab(tuple3 rgb) { return xyz2lab(rgb2xyz(rgb)); }
tuple3 xyz2rgb(tuple3 xyz);
tuple3 lab2xyz(tuple3 xyz);
inline tuple3 lab2rgb(tuple3 lab) { return xyz2rgb(lab2xyz(lab)); }
float labdist(tuple3 lab1, tuple3 lab2);