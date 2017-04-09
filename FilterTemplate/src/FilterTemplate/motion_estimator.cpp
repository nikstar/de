#include <unordered_map>

#include "motion_estimator.hpp"
#include "mat.h"

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

	if (use_half_pixel) {
		prev_map.emplace(ShiftDir::UP, prev_Y_up);
		prev_map.emplace(ShiftDir::LEFT, prev_Y_left);
		prev_map.emplace(ShiftDir::UPLEFT, prev_Y_upleft);
	}

	for (int i = 0; i < num_blocks_vert; ++i) {
		for (int j = 0; j < num_blocks_hor; ++j) {
			const auto block_id = i * num_blocks_hor + j;
			const auto hor_offset = j * BLOCK_SIZE;
			const auto vert_offset = first_row_offset + i * BLOCK_SIZE * width_ext;
			const auto cur = cur_Y + vert_offset + hor_offset;

			MV best_vector;
			best_vector.error = std::numeric_limits<long>::max();

			// Brute force
			for (const auto& prev_pair : prev_map) {
				const auto prev = prev_pair.second + vert_offset + hor_offset;

				for (int y = -BORDER; y <= BORDER; ++y) {
					for (int x = -BORDER; x <= BORDER; ++x) {
						const auto comp = prev + y * width_ext + x;
						const auto error = GetErrorSAD_16x16(cur, comp, width_ext);

						if (error < best_vector.error) {
							best_vector.x = x;
							best_vector.y = y;
							best_vector.shift_dir = prev_pair.first;
							best_vector.error = error;
						}
					}
				}
			}

			// Split into four subvectors if the error is too large
			if (best_vector.error > 1000) {
				best_vector.Split();

				for (int h = 0; h < 4; ++h) {
					auto& subvector = best_vector.SubVector(h);
					subvector.error = std::numeric_limits<long>::max();

					const auto hor_offset = j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0);
					const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0)) * width_ext;
					const auto cur = cur_Y + vert_offset + hor_offset;

					for (const auto& prev_pair : prev_map) {
						const auto prev = prev_pair.second + vert_offset + hor_offset;

						for (int y = -BORDER; y <= BORDER; ++y) {
							for (int x = -BORDER; x <= BORDER; ++x) {
								const auto comp = prev + y * width_ext + x;
								const auto error = GetErrorSAD_8x8(cur, comp, width_ext);

								if (error < subvector.error) {
									subvector.x = x;
									subvector.y = y;
									subvector.shift_dir = prev_pair.first;
									subvector.error = error;
								}
							}
						}
					}
				}

				if (best_vector.SubVector(0).error
					+ best_vector.SubVector(1).error
					+ best_vector.SubVector(2).error
					+ best_vector.SubVector(3).error > best_vector.error * 0.7)
					best_vector.Unsplit();
			}

			mvectors[block_id] = best_vector;
		}
	}
}


inline void update_best(MV& best, const MV& mv) {
	if (mv.error < best.error) {
		best = mv;
	}
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
				
			MV best_vector, current;
			// split for 8x8
			best_vector.Split();
			
			for (int h = 0; h < 4; ++h) {
				auto& best = best_vector.SubVector(h);
				best.error = std::numeric_limits<long>::max();

				const auto hor_offset = j * BLOCK_SIZE + ((h & 1) ? BLOCK_SIZE / 2 : 0);
				const auto vert_offset = first_row_offset + (i * BLOCK_SIZE + ((h > 1) ? BLOCK_SIZE / 2 : 0)) * width_ext;
				const auto cur = cur_Y + vert_offset + hor_offset;
				const auto prev = prev_Y + vert_offset + hor_offset;

				const uint8_t *comp = prev;

 				// check center (ZMP)
				SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
				update_best(best, current);
				if (best.error < zmp_threshold) { 
					mvectors[block_id] = best; 
					predicted = best; 
					continue; 
				}

				// Initial search
				const auto arm_length = j == 0 ? 2 : std::max(abs(predicted.x), abs(predicted.y));

				if (arm_length == 0) {
					// only search center
				}
				else {
					// serach four rood points
					// 1
					current.x = -arm_length;
					comp = prev - arm_length;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 2
					current.x = arm_length;
					comp = prev + arm_length;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 3
					current.x = 0;
					current.y = -arm_length;
					comp = prev - arm_length * width_ext;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 4
					current.y = arm_length;
					comp = prev + arm_length * width_ext;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);

					// also search predicted MV
					if (predicted.x != 0 && predicted.y != 0) {
						const auto comp = prev + predicted.y * width_ext + predicted.x;
						SafeSAD_8x8(predicted, cur, comp, width_ext, prev_Y); // fixme
						update_best(best, predicted);
					}
				}

				if (best.error < first_threshold) { 
					mvectors[block_id] = best; 
					predicted = best; 
					continue; 
				}
				
				// Local search (URP)
				do {
					current = best;
					const auto base = prev + current.y * width_ext + current.x;
					// 1
					current.x -= 1;
					comp = base - 1;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 2
					current.x += 2;
					comp = base + 1;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 3
					current.x -= 1;
					current.y -= 1;
					comp = base - width_ext;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					// 4
					current.y += 2;
					comp = base + width_ext;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y);
					update_best(best, current);
					current.y -= 1;
				} while (!(best.error < first_threshold) && (current.x != best.x || current.y != best.y));

				if (use_half_pixel && best.error > second_threshold) {
					current = best;
					auto ofs = vert_offset + hor_offset + current.y * width_ext + current.x;
					// 1
					current.shift_dir = ShiftDir::LEFT;
					comp = prev_Y_left + ofs;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_left);
					update_best(best, current);
					// 2
					current.shift_dir = ShiftDir::LEFT;
					current.x += 1;
					comp = prev_Y_left + ofs + 1;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_left);
					update_best(best, current);
					current.x -= 1;
					// 3
					current.shift_dir = ShiftDir::UP;
					comp = prev_Y_up + ofs;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_up);
					update_best(best, current);
					// 4
					current.shift_dir = ShiftDir::UP;
					current.y += 1;
					comp = prev_Y_up + ofs + width_ext;
					SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_up);
					update_best(best, current);
					current.y -= 1;

					if (best.shift_dir == ShiftDir::UP) {
						current = best;
						// 1
						current.shift_dir = ShiftDir::UPLEFT;
						comp = prev_Y_upleft + ofs;
						SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_upleft);
						update_best(best, current);
						// 2
						current.shift_dir = ShiftDir::UPLEFT;
						current.x += 1;
						comp = prev_Y_upleft + ofs + 1;
						SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_upleft);
						update_best(best, current);

					}

					if (best.shift_dir == ShiftDir::LEFT) {
						current = best;
						// 3
						current.shift_dir = ShiftDir::UPLEFT;
						comp = prev_Y_upleft + ofs;
						SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_upleft);
						update_best(best, current);
						// 4
						current.shift_dir = ShiftDir::UPLEFT;
						current.y += 1;
						comp = prev_Y_upleft + ofs + width_ext;
						SafeSAD_8x8(current, cur, comp, width_ext, prev_Y_upleft);
						update_best(best, current);
					}
				}

				predicted = best;
			}
			
			mvectors[block_id] = best_vector;
		}
	}
}
