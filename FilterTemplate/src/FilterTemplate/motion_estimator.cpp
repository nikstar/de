#include <unordered_map>

#include "motion_estimator.hpp"
#include "mat.h"

const int thresholds[5][3][3] = {
	{
		{ 768, 768, 512 },
		{ 256, 256, 192 },
		{ 256, 256, 192 }
	},
	{
		{ 768, 768, 512 },
		{ 256, 256, 192 },
		{ 256, 256, 192 }
	},
	{
		{ 768, 768, 512 },
		{ 256, 256, 192 },
		{ 256, 256, 192 }
	},
	{
		{ 768, 768, 512 },
		{ 256, 256, 192 },
		{ 256, 256, 192 }
	},
	{
		{ 768, 768, 512 },
		{ 256, 256, 192 },
		{ 256, 256, 192 }
	}
};


MotionEstimator::MotionEstimator(int width, int height, uint8_t quality, bool use_half_pixel)
	: width(width)
	, height(height)
	, quality(quality)
	, use_half_pixel(use_half_pixel)
	, width_ext(width + 2 * BORDER)
	, num_blocks_hor((width + BLOCK_SIZE - 1) / BLOCK_SIZE)
	, num_blocks_vert((height + BLOCK_SIZE - 1) / BLOCK_SIZE)
	, first_row_offset(width_ext * BORDER + BORDER)
{

	if (quality > 90) {
		zmp_threshold = (!use_half_pixel) ? 256 : 128;
		first_threshold = (!use_half_pixel) ? 256 : 128;
		second_threshold = 64;
	}
	else if (quality > 70) {
		zmp_threshold = 512;
		first_threshold = 512;
		second_threshold = 256;
	}
	else if (quality > 50) {
		zmp_threshold = 768;
		first_threshold = 768;
		second_threshold = 512;
	}
	else if (quality > 30) {
		zmp_threshold = 1024;
		first_threshold = 1024;
		second_threshold = 768;
	}
	else {
		zmp_threshold = 1536;
		first_threshold = 1536;
		second_threshold = 1024;
	}
	second_threshold = first_threshold;

	zmp_threshold /= 4;
	first_threshold /= 4;
	second_threshold /= 4;

	img_size = width_ext * height;
}

MotionEstimator::~MotionEstimator() {
}

void MotionEstimator::Estimate(const uint8_t* cur_Y,
	const uint8_t* prev_Y,
	const uint8_t* prev_Y_up,
	const uint8_t* prev_Y_left,
	const uint8_t* prev_Y_upleft,
	MV* mvectors) {
	//FullSearch(cur_Y, prev_Y, prev_Y_up, prev_Y_left, prev_Y_upleft, mvectors);
	ARPS(cur_Y, prev_Y, prev_Y_up, prev_Y_left, prev_Y_upleft, mvectors);
}

void MotionEstimator::FullSearch(const uint8_t* cur_Y,
	const uint8_t* prev_Y,
	const uint8_t* prev_Y_up,
	const uint8_t* prev_Y_left,
	const uint8_t* prev_Y_upleft,
	MV* mvectors)
{
	std::unordered_map<ShiftDir, const uint8_t*> prev_map{
		{ ShiftDir::NONE, prev_Y }
	};

	/*if (use_half_pixel) {
		prev_map.emplace(ShiftDir::UP, prev_Y_up);
		prev_map.emplace(ShiftDir::LEFT, prev_Y_left);
		prev_map.emplace(ShiftDir::UPLEFT, prev_Y_upleft);
	}*/

	for (int i = 0; i < num_blocks_vert; ++i) {
		for (int j = 0; j < num_blocks_hor; ++j) {
			const auto block_id = i * num_blocks_hor + j;
			const auto hor_offset = j * BLOCK_SIZE;
			const auto vert_offset = first_row_offset + i * BLOCK_SIZE * width_ext;
			const auto cur = cur_Y + vert_offset + hor_offset;

			MV best_vector;
			best_vector.error = std::numeric_limits<long>::max();

			// Split into four subvectors if the error is too large
			best_vector.Split();

			for (int h = 0; h < 4; ++h) {
				auto& best8 = best_vector.SubVector(h);
				best8.error = std::numeric_limits<long>::max();
				best8.Split();

				for (int h2 = 0; h2 < 4; ++h2) {
					auto& best4 = best8.SubVector(h2);
					best4.error = std::numeric_limits<long>::max();

					const auto hor_offset = j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0) + ((h2 & 1) ? BLOCK_SIZE / 4 : 0);
					const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0) + ((h2 > 1) ? BLOCK_SIZE / 4 : 0)) * width_ext;
					const auto cur = cur_Y + vert_offset + hor_offset;
						
					for (const auto& prev_pair : prev_map) {
						const auto prev = prev_pair.second + vert_offset + hor_offset;

						for (int y = -6; y <= 6; ++y) {
							for (int x = -BORDER; x <= BORDER; ++x) {
								const auto comp = prev + y * width_ext + x;
									
								auto shifted1 = cur - 2 * width_ext - 2;
								auto shifted2 = comp - 2 * width_ext - 2;

									
								if (shifted2 < prev_pair.second || shifted2 > prev_pair.second + first_row_offset + img_size) {
									continue;
								}

								const auto error = GetErrorSAD_8x8(shifted1, shifted2, width_ext);
								if (error < best4.error) {
									best4.x = x;
									best4.y = y;
									best4.shift_dir = prev_pair.first;
									best4.error = error;
								}
							}
						}
					}
				}

				best8.x = best8.SubVector(0).x;
				best8.y = best8.SubVector(0).y;

			}
		
			mvectors[block_id] = best_vector;
		}
	}
}


void SafeSAD_16x16(MV& mv, const uint8_t *block1, const uint8_t *block2, const int stride, const uint8_t *prev_Y, const int first_row_offset, const int img_size) {
	if (block2 < prev_Y + first_row_offset || block2 > prev_Y + first_row_offset + img_size) {
		mv.error = std::numeric_limits<long>::max();
		return;
	}
	mv.error = GetErrorSAD_16x16(block1, block2, stride);
}

void SafeSAD_8x8(MV& mv, const uint8_t *block1, const uint8_t *block2, const int stride, const uint8_t *prev_Y, const int first_row_offset, const int img_size) {
	if (block2 < prev_Y + first_row_offset || block2 > prev_Y + first_row_offset + img_size) {
		mv.error = std::numeric_limits<long>::max();
		return;
	}
	mv.error = GetErrorSAD_8x8(block1, block2, stride);
}

void SafeSAD_4x4(MV& mv, const uint8_t *block1, const uint8_t *block2, const int stride, const uint8_t *prev_Y, const int first_row_offset, const int img_size) {
	auto shifted1 = block1 - 2 * stride - 2;
	auto shifted2 = block2 - 2 * stride - 2;
	
	if (shifted2 < prev_Y || shifted2 > prev_Y + first_row_offset + img_size) {
		mv.error = std::numeric_limits<long>::max();
		return;
	}
	
	mv.error = GetErrorSAD_8x8(shifted1, shifted2, stride);
}


inline void update(MV& best, const MV& mv) {
	if (mv.error < best.error) {
		best = mv;
	}
}


template <void(*SAD)(MV&, const uint8_t *, const uint8_t *, const int, const uint8_t *, const int, const int)>
void MotionEstimator::EstimateAtLevel(bool at_edge, const uint8_t *prev_Y, const uint8_t *cur, const uint8_t *prev, MV& predicted, MV& best) {
	auto comp = prev;
	MV current;

	// check center (ZMP)
	SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
	update(best, current);

	if (best.error < zmp_threshold) {
		return;
	}

	// Initial search
	const auto arm_length = at_edge ? 2 : std::max(abs(predicted.x), abs(predicted.y));

	if (arm_length == 0) {
		// only search center
	}
	else {
		// serach four rood points
		// 1
		current.x = -arm_length;
		comp = prev - arm_length;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 2
		current.x = arm_length;
		comp = prev + arm_length;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 3
		current.x = 0;
		current.y = -arm_length;
		comp = prev - arm_length * width_ext;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 4
		current.y = arm_length;
		comp = prev + arm_length * width_ext;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);

		// also search predicted MV
		if (!at_edge && predicted.x != 0 && predicted.y != 0) {
			const auto comp = prev + predicted.y * width_ext + predicted.x;
			SAD(predicted, cur, comp, width_ext, prev_Y, first_row_offset, img_size); // fixme
			update(best, predicted);
		}
	}

	if (best.error < first_threshold) {
		return;
	}

	// Local search (URP)
	do {
		current = best;
		const auto base = prev + current.y * width_ext + current.x;
		// 1
		current.x -= 1;
		comp = base - 1;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 2
		current.x += 2;
		comp = base + 1;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 3
		current.x -= 1;
		current.y -= 1;
		comp = base - width_ext;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		// 4
		current.y += 2;
		comp = base + width_ext;
		SAD(current, cur, comp, width_ext, prev_Y, first_row_offset, img_size);
		update(best, current);
		current.y -= 1;
	} while (!(best.error < first_threshold) && (current.x != best.x || current.y != best.y));

	/*if (use_half_pixel && best.error > second_threshold) {
		current = best;
		auto ofs = vert_offset + hor_offset + current.y * width_ext + current.x;
		// 1
		current.shift_dir = ShiftDir::LEFT;
		comp = prev_Y_left + ofs;
		SAD(current, cur, comp, width_ext, prev_Y_left);
		update(best, current);
		// 2
		current.shift_dir = ShiftDir::LEFT;
		current.x += 1;
		comp = prev_Y_left + ofs + 1;
		SAD(current, cur, comp, width_ext, prev_Y_left);
		update(best, current);
		current.x -= 1;
		// 3
		current.shift_dir = ShiftDir::UP;
		comp = prev_Y_up + ofs;
		SAD(current, cur, comp, width_ext, prev_Y_up);
		update(best, current);
		// 4
		current.shift_dir = ShiftDir::UP;
		current.y += 1;
		comp = prev_Y_up + ofs + width_ext;
		SAD(current, cur, comp, width_ext, prev_Y_up);
		update(best, current);
		current.y -= 1;

		if (best.shift_dir == ShiftDir::UP) {
			current = best;
			// 1
			current.shift_dir = ShiftDir::UPLEFT;
			comp = prev_Y_upleft + ofs;
			SAD(current, cur, comp, width_ext, prev_Y_upleft);
			update(best, current);
			// 2
			current.shift_dir = ShiftDir::UPLEFT;
			current.x += 1;
			comp = prev_Y_upleft + ofs + 1;
			SAD(current, cur, comp, width_ext, prev_Y_upleft);
			update(best, current);

		}

		if (best.shift_dir == ShiftDir::LEFT) {
			current = best;
			// 3
			current.shift_dir = ShiftDir::UPLEFT;
			comp = prev_Y_upleft + ofs;
			SAD(current, cur, comp, width_ext, prev_Y_upleft);
			update(best, current);
			// 4
			current.shift_dir = ShiftDir::UPLEFT;
			current.y += 1;
			comp = prev_Y_upleft + ofs + width_ext;
			SAD(current, cur, comp, width_ext, prev_Y_upleft);
			update(best, current);
		}
	}*/
}

void MotionEstimator::ARPS(const uint8_t* cur_Y,
	const uint8_t* prev_Y,
	const uint8_t* prev_Y_up,
	const uint8_t* prev_Y_left,
	const uint8_t* prev_Y_upleft,
	MV* mvectors) 
{
	// Uses MV of the left block as estimation
	MV predicted;

	for (int i = 0; i < num_blocks_vert; ++i) {
		for (int j = 0; j < num_blocks_hor; ++j) {
			const auto block_id = i * num_blocks_hor + j;
				
			MV best16;
			best16.Split(); // always split for 8x8
			
			for (int h = 0; h < 4; ++h) {
				auto& best8 = best16.SubVector(h);
				best8.error = std::numeric_limits<long>::max();

				const auto hor_offset =                      j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0);
				const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0)) * width_ext;
				const auto cur = cur_Y + vert_offset + hor_offset;
				const auto prev = prev_Y + vert_offset + hor_offset;
		
				const auto at_edge = j == 0 && (h & 1) == 0;
				
				EstimateAtLevel<&(SafeSAD_8x8)>(at_edge, prev_Y, cur, prev, predicted, best8);
				
				if (best8.error > -1) { // was 250
					best8.Split();

					predicted = best8;

					for (int h2 = 0; h2 < 4; ++h2) {
						auto& best4 = best8.SubVector(h2);
						best4.error = std::numeric_limits<long>::max();

						const auto hor_offset =                      j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0) + ((h2 & 1) ? BLOCK_SIZE / 4 : 0);
						const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0) + ((h2 > 1) ? BLOCK_SIZE / 4 : 0)) * width_ext;
						const auto cur = cur_Y + vert_offset + hor_offset;
						const auto prev = prev_Y + vert_offset + hor_offset;

						const auto at_edge = j == 0 && (h & 1) == 0 && (h2 & 1) == 0;

						EstimateAtLevel<&(SafeSAD_4x4)>(at_edge, prev_Y, cur, prev, predicted, best4); // FIX thresholds
					}

					/*if (best8.SubVector(0).error + best8.SubVector(1).error + best8.SubVector(2).error + best8.SubVector(3).error >= 3 * best8.error) {
						best8.Unsplit(); 
					}*/
				}

				predicted = best8;
			}
			
			mvectors[block_id] = best16;
		}
	}
}
