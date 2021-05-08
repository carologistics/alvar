/*
 * This file is part of ALVAR, A Library for Virtual and Augmented Reality.
 *
 * Copyright 2007-2012 VTT Technical Research Centre of Finland
 *
 * Contact: VTT Augmented Reality Team <alvar.info@vtt.fi>
 *          <http://www.vtt.fi/multimedia/alvar.html>
 *
 * ALVAR is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ALVAR; if not, see
 * <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.
 */

#include "TrackerFeatures.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

namespace alvar {
using namespace std;

TrackerFeatures::TrackerFeatures(int    _max_features,
                                 int    _min_features,
                                 double _quality_level,
                                 double _min_distance,
                                 int    _pyr_levels,
                                 int    _win_size)
: x_res(0),
  y_res(0),
  frame_count(0),
  quality_level(0),
  min_distance(0),
  min_features(0),
  max_features(0),
  status(),
  img_eig(),
  img_tmp(),
  gray(),
  prev_gray(),
  pyramid(),
  prev_pyramid(),
  mask(),
  next_id(0),
  win_size(0),
  pyr_levels(0),
  prev_features(),
  features(),
  prev_feature_count(0),
  feature_count(0),
  prev_ids(0),
  ids(0)
{
	next_id    = 1; // When this should be reset?
	pyr_levels = _pyr_levels;
	win_size   = _win_size;
	ChangeSettings(_max_features, _min_features, _quality_level, _min_distance);
}

TrackerFeatures::~TrackerFeatures()
{
	if (prev_ids)
		delete[] prev_ids;
	if (ids)
		delete[] ids;
	prev_features.clear();
	features.clear();
	img_eig.release();
	img_tmp.release();
	gray.release();
	prev_gray.release();
	pyramid.release();
	prev_pyramid.release();
	mask.release();
	status.release();
}

void
TrackerFeatures::ChangeSettings(int    _max_features,
                                int    _min_features,
                                double _quality_level,
                                double _min_distance)
{
	if (_max_features == max_features && _min_features == min_features
	    && _quality_level == quality_level && _min_distance == min_distance)
		return;

	int common_features = min(feature_count, _max_features);
	max_features        = _max_features;
	min_features        = _min_features;
	quality_level       = _quality_level;
	min_distance        = _min_distance;
	status.release();
	if (prev_ids)
		delete[] prev_ids;
	prev_ids = NULL;
	prev_features.clear();
	if (ids) {
		int *ids_new = new int[max_features];
		assert(common_features < max_features);
		memcpy(ids_new, ids, sizeof(int) * common_features);
		delete[] ids;
		ids = ids_new;
	} else {
		ids = new int[max_features];
	}
	if (!features.empty()) {
		std::vector<cv::Point2f> features_new = std::vector<cv::Point2f>(max_features);
		memcpy(&features_new, &features, sizeof(std::vector<cv::Point2f>) * common_features);
		features.clear();
		features = features_new;
	} else {
		features = std::vector<cv::Point2f>(max_features);
	}
	status             = cv::Mat();
	prev_ids           = new int[max_features];
	prev_features      = std::vector<cv::Point2f>(max_features);
	prev_feature_count = 0;
	feature_count      = common_features;

	assert(ids);
	assert(prev_ids);
}

void
TrackerFeatures::Reset()
{
	feature_count = 0;
	frame_count   = 0;
}

bool
TrackerFeatures::DelFeature(int index)
{
	if (index > feature_count)
		return false;
	feature_count--;
	for (int i = index; i < feature_count; i++) {
		features[i] = features[i + 1];
		ids[i]      = ids[i + 1];
	}
	return true;
}

bool
TrackerFeatures::DelFeatureId(int id)
{
	for (int i = 0; i < feature_count; i++) {
		if (ids[i] == id)
			return DelFeature(i);
	}
	return false;
}

int
TrackerFeatures::Purge()
{
	int   removed_count = 0;
	float dist          = 0.7f * float(min_distance);
	for (int i = 1; i < feature_count; i++) {
		for (int ii = 0; ii < i; ii++) {
			float dx = features[i].x - features[ii].x;
			float dy = features[i].y - features[ii].y;
			if (dx < 0)
				dx = -dx;
			if (dy < 0)
				dy = -dy;
			if ((dx < dist) && (dy < dist)) {
				DelFeature(i);
				i--;
				removed_count++;
				break;
			}
		}
	}
	return removed_count;
}

double
TrackerFeatures::TrackHid(const cv::Mat &img, cv::Mat &new_features_mask, bool add_features)
{
	if ((x_res != img.cols) || (y_res != img.rows)) {
		img_eig.release();
		img_tmp.release();
		gray.release();
		prev_gray.release();
		pyramid.release();
		prev_pyramid.release();
		mask.release();
		x_res        = img.cols;
		y_res        = img.rows;
		img_eig      = cv::Mat(cv::Size(img.cols, img.rows), CV_MAKETYPE(CV_32F, 1));
		img_tmp      = cv::Mat(cv::Size(img.cols, img.rows), CV_MAKETYPE(CV_32F, 1));
		gray         = cv::Mat(cv::Size(img.cols, img.rows), CV_MAKETYPE(CV_8U, 1));
		prev_gray    = cv::Mat(cv::Size(img.cols, img.rows), CV_MAKETYPE(CV_8U, 1));
		pyramid      = cv::Mat(cv::Size(img.cols + 8, img.rows / 3), CV_MAKETYPE(CV_8U, 1));
		prev_pyramid = cv::Mat(cv::Size(img.cols + 8, img.rows / 3), CV_MAKETYPE(CV_8U, 1));
		mask         = cv::Mat(cv::Size(img.cols, img.rows), CV_MAKETYPE(CV_8U, 1));
		frame_count  = 0;
		if (min_distance == 0) {
			min_distance = std::sqrt(double(img.cols * img.rows / max_features));
			min_distance *= 0.8; //(double(min_features)/max_features);
		}
	}
	// Swap
	cv::swap(prev_gray, gray);
	cv::swap(prev_features, features);
	prev_feature_count = feature_count;
	memcpy(prev_ids, ids, sizeof(int) * max_features);
	if (img.channels() == 1) {
		img.copyTo(gray);
	} else {
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
	}
	// TODO: We used to add features here
	//if (prev_feature_count < 1) return -1;
	frame_count++;
	if (frame_count <= 1) {
		memcpy(&features, &prev_features, sizeof(cv::Mat) * prev_feature_count);
		memcpy(ids, prev_ids, sizeof(int) * prev_feature_count);
		feature_count = prev_feature_count;
	} else if (prev_feature_count > 0) {
		// Track, no clue whether this is ported correctly -Sebastian
		cv::calcOpticalFlowPyrLK(prev_gray,
		                         gray,
		                         prev_pyramid,
		                         pyramid,
		                         status,
		                         cv::Mat(),
		                         cv::Size(win_size, win_size),
		                         pyr_levels,
		                         cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
		                                          20,
		                                          0.03),
		                         0);
		feature_count = 0;
		for (int i = 0; i < prev_feature_count; i++) {
			if (!status.at<bool>(i, 0)) //this is questionable, no clue on indexing here - Sebastian
				continue;
			features[feature_count] = features[i];
			ids[feature_count]      = prev_ids[i];
			feature_count++;
		}
	}

	if (add_features)
		AddFeatures(new_features_mask);

	return 1;
}

double
TrackerFeatures::Reset(const cv::Mat &img, cv::Mat &new_features_mask)
{
	feature_count = 0;
	frame_count   = 0;
	return TrackHid(img, new_features_mask);
}

double
TrackerFeatures::Track(cv::Mat &img, bool add_features)
{
	return TrackHid(img, mask = cv::Mat()); //, add_features);
}

double
TrackerFeatures::Track(cv::Mat &img, cv::Mat &mask)
{
	return TrackHid(img, mask); //, true);
}

cv::Mat
TrackerFeatures::NewFeatureMask()
{
	cv::Mat mask = cv::Mat();
	mask         = cv::Scalar(255);
	for (int i = 0; i < feature_count; i++) {
		cv::rectangle(mask,
		              cv::Point(int(features[i].x - min_distance), int(features[i].y - min_distance)),
		              cv::Point(int(features[i].x + min_distance), int(features[i].y + min_distance)),
		              cv::Scalar(0),
		              cv::FILLED);
	}
	return mask;
}

int
TrackerFeatures::AddFeatures(cv::Mat &new_features_mask)
{
	if (gray.empty())
		return 0;
	if (feature_count < min_features) {
		int new_feature_count = max_features - feature_count;
		if (new_features_mask.empty()) {
			cv::Mat mask = cv::Mat();
			mask         = cv::Scalar(255);
			for (int i = 0; i < feature_count; i++) {
				cv::rectangle(mask,
				              cv::Point(int(features[i].x - min_distance),
				                        int(features[i].y - min_distance)),
				              cv::Point(int(features[i].x + min_distance),
				                        int(features[i].y + min_distance)),
				              cv::Scalar(0),
				              cv::FILLED);
			}
			// Find new features
			cv::goodFeaturesToTrack(
			  gray, features, new_feature_count, quality_level, min_distance, mask, 3, true, 0.04);

		} else {
			cv::goodFeaturesToTrack(gray,
			                        features,
			                        new_feature_count,
			                        quality_level,
			                        min_distance,
			                        new_features_mask,
			                        3,
			                        true,
			                        0.04);
		}
		if (new_feature_count >= 1) {
			for (int i = feature_count; i < feature_count + new_feature_count; i++) {
				ids[i]  = next_id;
				next_id = ((next_id + 1) % (0x7fff));
			}
			feature_count += new_feature_count;
		}
	}
	return feature_count;
}

} // namespace alvar
