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

#ifndef TRACKERFEATURES_H
#define TRACKERFEATURES_H

/**
 * \file TrackerFeatures.h
 *
 * \brief This file implements a feature tracker.
 */

#include "Tracker.h"

namespace alvar {

/**
 * \brief \e TrackerFeatures tracks features using OpenCV's cvGoodFeaturesToTrack and cvCalcOpticalFlowPyrLK
 */
class ALVAR_EXPORT TrackerFeatures : public Tracker
{
protected:
	int     x_res, y_res;
	int     frame_count;
	double  quality_level;
	double  min_distance;
	int     min_features;
	int     max_features;
	cv::Mat status;
	cv::Mat img_eig;
	cv::Mat img_tmp;
	cv::Mat gray;
	cv::Mat prev_gray;
	cv::Mat pyramid;
	cv::Mat prev_pyramid;
	cv::Mat mask;
	int     next_id;
	int     win_size;
	int     pyr_levels;

	/** \brief Reset track features on specified mask area */
	double TrackHid(const cv::Mat &img, cv::Mat &mask, bool add_features = true);

public:
	/** \brief \e Track result: previous features */
	std::vector<cv::Point2f>
	  prev_features; // this used to be an array of points now, its an array of Matrices? -Sebastian
	                 /** \brief \e Track result: current features */
	std::vector<cv::Point2f>
	  features; // this used to be an array of points now, its an array of Matrices? -Sebastian
	/** \brief \e Track result: count of previous features */
	int prev_feature_count;
	/** \brief \e Track result: count of current features */
	int feature_count;
	/** \brief \e Track result: ID:s for previous features */
	int *prev_ids;
	/** \brief \e Track result: ID:s for current features */
	int *ids;
	/**
	 * \brief Constructor for \e TrackerFeatures tracks features using OpenCV's cvGoodFeaturesToTrack and cvCalcOpticalFlowPyrLK
	 * \param _max_features The maximum amount of features to be tracked
	 * \param _min_features The minimum amount of features. The featureset is filled up when the number of features is lower than this.
	 * \param _quality_level Multiplier for the maxmin eigenvalue; specifies minimal accepted quality of image corners.
	 * \param _min_distance Limit, specifying minimum possible distance between returned corners; Euclidian distance is used. 
	 *         If 0 given we use default value aiming for uniform cover: _min_distance = 0.8*sqrt(x_res*y_res/max_features))
     * \param _pyr_levels Number of pyramid levels
	 */
	TrackerFeatures(int    _max_features  = 100,
	                int    _min_features  = 90,
	                double _quality_level = 0.01,
	                double _min_distance  = 10,
	                int    _pyr_levels    = 1,
	                int    _win_size      = 3);
	/** \brief Destructor */
	~TrackerFeatures();
	/** \brief Change settings while running */
	void ChangeSettings(int    _max_features  = 100,
	                    int    _min_features  = 90,
	                    double _quality_level = 0.01,
	                    double _min_distance  = 10);
	/** \brief Reset */
	void Reset();
	/** \brief Reset track features on specified mask area */
	double Reset(const cv::Mat &img, cv::Mat &mask);
	/** \brief Stop tracking the identified feature (with index for features array)*/
	bool DelFeature(int index);
	/** \brief Stop tracking the identified feature (with feature id) */
	bool DelFeatureId(int id);
	/** \brief Track features */
	double
	Track(cv::Mat &img)
	{
		return Track(img, true);
	}
	/** \brief Track features */
	double Track(cv::Mat &img, bool add_features);
	/** \brief Track features */
	double Track(cv::Mat &img, cv::Mat &mask);
	/** \brief add features to the previously tracked frame if there are less than min_features */
	int AddFeatures(cv::Mat &mask);
	/** \brief Create and get the pointer to new_features_mask */
	cv::Mat NewFeatureMask();
	/** \brief Purge features that are considerably closer than the defined min_distance. 
	 *
	 * Note, that we always try to maintain the smaller id's assuming that they are older ones
	 */
	int Purge();
};

} // namespace alvar

#endif
