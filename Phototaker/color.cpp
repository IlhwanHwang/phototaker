#include "color.h"
#include <cmath>

tuple3 rgb2xyz(tuple3 rgb) {
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;

	if (r > 0.04045)
		r = pow((r + 0.055) / 1.055, 2.4);
	else
		r = r / 12.92;

	if (g > 0.04045)
		g = pow((g + 0.055) / 1.055, 2.4);
	else
		g = g / 12.92;
			
	if (b > 0.04045)
		b = pow((b + 0.055) / 1.055, 2.4);
	else
		b = b / 12.92;

	r *= 100.0;
	g *= 100.0;
	b *= 100.0;

	float x, y, z;
	//Observer. = 2¡Æ, Illuminant = D65
	x = r * 0.4124 + g * 0.3576 + b * 0.1805;
	y = r * 0.2126 + g * 0.7152 + b * 0.0722;
	z = r * 0.0193 + g * 0.1192 + b * 0.9505;

	return tuple3(x, y, z);
}

tuple3 xyz2lab(tuple3 xyz) {
	float x = xyz.x;
	float y = xyz.y;
	float z = xyz.z;
	
	x = x / 95.047;        //ref_X =  95.047   Observer= 2¡Æ, Illuminant= D65
	y = y / 100.000;          //ref_Y = 100.000
	z = z / 108.883;          //ref_Z = 108.883

	if (x > 0.008856)
		x = pow(x, 1.0 / 3.0);
	else
		x = (7.787 * x) + (16.0 / 116.0);
	
	if (y > 0.008856)
		y = pow(y, 1.0 / 3.0);
	else
		y = (7.787 * y) + (16.0 / 116.0);

	if (z > 0.008856)
		z = pow(z, 1.0 / 3.0);
	else
		z = (7.787 * z) + (16.0 / 116.0);

	float l, a, b;
	l = (116.0 * y) - 16.0;
	a = 500.0 * (x - y);
	b = 200.0 * (y - z);

	return tuple3(l, a, b);
}

tuple3 xyz2rgb(tuple3 xyz) {
	float x = xyz.x;
	float y = xyz.y;
	float z = xyz.z;

	x = x / 100.0;        //X from 0 to  95.047      (Observer = 2¡Æ, Illuminant = D65)
	y = y / 100.0;        //Y from 0 to 100.000
	z = z / 100.0;        //Z from 0 to 108.883

	float r, g, b;
	r = x *  3.2406 + y * -1.5372 + z * -0.4986;
	g = x * -0.9689 + y *  1.8758 + z *  0.0415;
	b = x *  0.0557 + y * -0.2040 + z *  1.0570;

	if (r > 0.0031308)
		r = 1.055 * pow(r, 1.0 / 2.4) - 0.055;
	else
		r = 12.92 * r;

	if (g > 0.0031308)
		g = 1.055 * pow(g, 1.0 / 2.4) - 0.055;
	else
		g = 12.92 * g;

	if (b > 0.0031308)
		b = 1.055 * pow(b, 1.0 / 2.4) - 0.055;
	else
		b = 12.92 * b;

	return tuple3(r, g, b);
}

tuple3 lab2xyz(tuple3 lab) {
	float l = lab.x;
	float a = lab.y;
	float b = lab.z;
	float x, y, z;

	y = (l + 16.0) / 116.0;
	x = a / 500.0 + y;
	z = y - b / 200.0;

	if (pow(y, 3.0) > 0.008856)
		y = pow(y, 3.0);
	else
		y = (y - 16.0 / 116.0) / 7.787;

	if (pow(x, 3.0) > 0.008856)
		x = pow(x, 3.0);
	else
		x = (x - 16.0 / 116.0) / 7.787;

	if (pow(z, 3.0) > 0.008856)
		z = pow(z, 3.0);
	else
		z = (z - 16.0 / 116.0) / 7.787;

	x = 95.047 * x;     //ref_X =  95.047     Observer= 2¡Æ, Illuminant= D65
	y = 100.000 * y;    //ref_Y = 100.000
	z = 108.883 * z;     //ref_Z = 108.883

	return tuple3(x, y, z);
}

float labdist(tuple3 lab1, tuple3 lab2) {
	return sqrt(pow(lab1.x - lab2.x, 2.0) + pow(lab1.y - lab2.y, 2.0) + pow(lab1.z - lab2.z, 2.0));
}