#pragma once

#include <exception>

class Mat {
	uint8_t *data;
public:
	int m, n;
	int width;

	Mat(int m, int n, uint8_t *data, int width = 0) : m(m), n(n), data(data), width(width > 0 ? width : n) {}
	~Mat() {} // note underlying data structure is not deleted
	
	inline Mat cropped(int origin_x, int origin_y, int width, int height) const {
		if (origin_y + height > m || origin_x + width > n) throw std::exception("Cannot crop");
		return Mat(height, width, data + origin_y*width + origin_x, width);
	}

	inline Mat& shift(int x, int y) {
		// note: no bounds check here
		data += y*width + x;
	}

	inline uint8_t operator()(int x, int y) const {
		if (y >= m || x >= n) throw std::exception("Index out of bounds");
		return data[y*width + x];
	}

	inline uint8_t& operator()(int x, int y) {
		if (y >= m || x >= n) throw std::exception("Index out of bounds");
		return data[y*width + x];
	}
};