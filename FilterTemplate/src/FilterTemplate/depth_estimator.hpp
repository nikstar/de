#pragma once

#include <cstdint>
#include <deque>
#include "mv.hpp"

class DepthEstimator {
public:
	/// Constructor
	DepthEstimator(int width, int height, uint8_t quality);

	/// Destructor
	~DepthEstimator();

	/// Copy constructor (deleted)
	DepthEstimator(const DepthEstimator&) = delete;

	/// Move constructor
	DepthEstimator(DepthEstimator&&) = default;

	/// Copy assignment (deleted)
	DepthEstimator& operator=(const DepthEstimator&) = delete;

	/// Move assignment
	DepthEstimator& operator=(DepthEstimator&&) = default;

	/**
	 * Estimate the depth map of a frame
	 *
	 * @param[in] cur_Y array of pixel Y values of the current frame
	 * @param[in] cur_U array of pixel U values of the current frame
	 * @param[in] cur_V array of pixel V values of the current frame
	 * @param[in] mvectors array of motion vectors
	 * @param[out] depth_map output array of pixel depth values
	 */
	void Estimate(const uint8_t* cur_Y,
	              const int16_t* cur_U,
	              const int16_t* cur_V,
	              const MV* mvectors,
	              uint8_t* depth_map);

private:
	/// Frame width (not including borders)
	const int width;

	/// Frame height (not including borders)
	const int height;

	/// Quality
	const uint8_t quality;

	/// Extended frame width (including borders)
	const int width_ext;

	/// Number of blocks per X-axis
	const int num_blocks_hor;

	/// Number of blocks per Y-axis
	const int num_blocks_vert;

	/// Position of the first pixel of the frame in the extended frame
	const int first_row_offset;

	// data
	const int max_history = 3;
	std::deque<uint8_t *> history;


	/// Convert MV into depth map
	void CreateInitialMap(const MV* mvectors, uint8_t* depth_map);
	
	/// Update history with new motion vectors
	void UpdateHistory(const MV* mvectors);


	/// Apply temporal median filter
	void ApplyMedianFilter(uint8_t* depth_map);

	/// Apply cross bilateral filter based on image data
	void ApplyCrossBilateralFilter(uint8_t *depth_map, const uint8_t * cur_Y, const int16_t * cur_U, const int16_t * cur_V);

	/// Cache DM for use in median filter
	void Cache(uint8_t * depth_map);

};
