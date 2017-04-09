#include <cmath>

#include "motion_estimator.hpp"
#include "depth_estimator.hpp"

//http://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel

double * GaussianFilter(int W, double sigma) {
	double *kernel = new double [W * W];
	double mean = W / 2.0;
	double sum = 0.0; // For accumulating the kernel values
	for (int x = 0; x < W; ++x) {
		for (int y = 0; y < W; ++y) {
			kernel[x + y * W] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
				/ (2 * 3.14159 * sigma * sigma);

			// Accumulate the kernel values
			sum += kernel[x + y * W];
		}
	}

	// Normalize the kernel
	for (int x = 0; x < W; ++x) {
		for (int y = 0; y < W; ++y) {
			kernel[x + y * W] /= sum;
		}
	}

	return kernel;
}



DepthEstimator::DepthEstimator(int width, int height, uint8_t quality)
	: width(width)
	, height(height)
	, quality(quality)
	, width_ext(width + 2 * MotionEstimator::BORDER)
	, num_blocks_hor((width + MotionEstimator::BLOCK_SIZE - 1) / MotionEstimator::BLOCK_SIZE)
	, num_blocks_vert((height + MotionEstimator::BLOCK_SIZE - 1) / MotionEstimator::BLOCK_SIZE)
	, first_row_offset(width_ext * MotionEstimator::BORDER + MotionEstimator::BORDER) {
	// PUT YOUR CODE HERE
}

DepthEstimator::~DepthEstimator() {
	// PUT YOUR CODE HERE
}




void DepthEstimator::Estimate(const uint8_t* cur_Y,
                              const int16_t* cur_U,
                              const int16_t* cur_V,
                              const MV* mvectors,
                              uint8_t* depth_map) {
	// PUT YOUR CODE HERE
	constexpr double MULTIPLIER = 32.0;

	double *filter = GaussianFilter(5, 1.0);

	// create initial map
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const auto i = (y / MotionEstimator::BLOCK_SIZE);
			const auto j = (x / MotionEstimator::BLOCK_SIZE);
			auto mv = mvectors[i * num_blocks_hor + j];

			// If the vector was split (so 8x8 instead of 16x16), use the correct subvector.
			// If you use 4x4 vectors, add another similar if statement inside.
			if (mv.IsSplit()) {
				const auto h = (((y % MotionEstimator::BLOCK_SIZE) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 2)
					+ (((x % MotionEstimator::BLOCK_SIZE) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 1);
				mv = mv.SubVector(h);
				
				if (mv.IsSplit()) {
					const auto h = (((y % (MotionEstimator::BLOCK_SIZE / 2)) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 2)
						+ (((x % (MotionEstimator::BLOCK_SIZE / 2)) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 1);
					mv = mv.SubVector(h);
				}
			}

			depth_map[y * width + x] = static_cast<uint8_t>(abs(mv.x) * MULTIPLIER);
		}
	}




}
