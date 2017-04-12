#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#include "motion_estimator.hpp"
#include "depth_estimator.hpp"

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
	CreateInitialMap(mvectors, depth_map);
	
	UpdateHistory(mvectors);
	ApplyCrossBilateralFilter(depth_map, cur_Y, cur_U, cur_V);
	ApplyMedianFilter(depth_map);
	Cache(depth_map);
}

void DepthEstimator::CreateInitialMap(const MV * mvectors, uint8_t * depth_map)
{
	constexpr int MULTIPLIER = 16.0;

	constexpr int block_size = 4; // FIXME: dependent on BLOCK_SIZE
	constexpr unsigned mask16 = MotionEstimator::BLOCK_SIZE - 1;
	constexpr unsigned mask8 = mask16 >> 1;
	constexpr unsigned mask4 = mask8 >> 1;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const auto i = y >> block_size;
			const auto j = x >> block_size;
			const auto block_id = i * num_blocks_hor + j;
			
			const auto h =
				(((y & mask16) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 2)
				+ (((x & mask16) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 1);
			const auto h2 =
				(((y & mask8) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 2)
				+ (((x & mask8) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 1);

			const auto & mv = mvectors[block_id].SubVector(h).SubVector(h2);
			
			depth_map[y * width + x] = static_cast<uint8_t>(std::min(abs(mv.x) * MULTIPLIER, 255));
		}
	}
}

void DepthEstimator::UpdateHistory(const MV * mvectors)
{
	auto prev = new uint8_t[height*width];

	constexpr int block_size = 4; // FIXME: dependent on BLOCK_SIZE
	constexpr unsigned mask16 = MotionEstimator::BLOCK_SIZE - 1;
	constexpr unsigned mask8 = mask16 >> 1;
	constexpr unsigned mask4 = mask8 >> 1;

	for (auto m : history) {
		memcpy(prev, m, height * width);
		
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const auto i = y >> block_size;
				const auto j = x >> block_size;
				const auto block_id = i * num_blocks_hor + j;
				
				const auto h =
					(((y & mask16) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 2)
					+ (((x & mask16) < (MotionEstimator::BLOCK_SIZE / 2)) ? 0 : 1);
				const auto h2 =
					(((y & mask8) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 2)
					+ (((x & mask8) < (MotionEstimator::BLOCK_SIZE / 4)) ? 0 : 1);

				const auto & mv = mvectors[block_id].SubVector(h).SubVector(h2);

				const auto prev_x = std::min(std::max(x + mv.x, 0), width - 1);
				const auto prev_y = std::min(std::max(y + mv.y, 0), height - 1);
				m[y * width + x] = prev[prev_y * width + prev_x];
			}
		}
	}

	delete[] prev;
}

void DepthEstimator::ApplyMedianFilter(uint8_t * depth_map)
{
	if (history.size() >= max_history) {
		return;
	}

	std::vector<uint8_t> v;
	v.reserve(max_history + 1);

	for (int i = 0; i < height*width; ++i) {
		// add relevant points to vector
		for (auto m : history) {
			v.push_back(m[i]);
		}
		v.push_back(depth_map[i]);
		// find median
		std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
		// update value
		depth_map[i] = v[v.size() / 2];
		v.clear();
	}
}

inline int sqr(int x) {
	return x * x;
}

void DepthEstimator::ApplyCrossBilateralFilter(uint8_t * depth_map, const uint8_t * cur_Y, const int16_t * cur_U, const int16_t * cur_V)
{
	constexpr int S = 3;
	constexpr int W = 2 * S + 1;
	constexpr double sigma1 = 15.0, sigma2 = 100.0;

	
	auto filt = new uint8_t[height*width];

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const auto ofs = y * width + x;

			double acc = 0.0;
			double sum = 0.0; // For accumulating the kernel values

			auto depth_center = &depth_map[ofs];
			auto Y_center = &cur_Y[y * width_ext + x];

			for (int i = std::max(-S, 0 - y); i < std::min(S, height - y - 1); ++i) {
				for (int j = std::max(-S, 0 - x); j < std::min(S, width - x - 1); ++j) {
					auto v = 
						exp(-0.5*sqrt(sqr(i)+sqr(j)) / sigma1) *
						exp(-0.5*abs(*Y_center - *(Y_center + width_ext * i + j)) / sigma2);

					acc += v * *(depth_center + i * width + j);
					sum += v;
				}
			}

			filt[ofs] = acc / sum;
		}
	}
	memcpy(depth_map, filt, height*width);

	delete[] filt;
}

void DepthEstimator::Cache(uint8_t * depth_map)
{
	if (history.size() >= max_history) {
		auto first = history.front();
		history.pop_front();
		delete[] first;
	}
	auto copy = new uint8_t[height * width];
	memcpy(copy, depth_map, sizeof(uint8_t) * height * width);
	history.push_back(copy);
}
