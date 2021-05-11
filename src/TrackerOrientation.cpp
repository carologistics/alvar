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

#include "TrackerOrientation.h"

#include "Optimization.h"

#include <opencv2/opencv.hpp>

using namespace std;

namespace alvar {
using namespace std;

void
TrackerOrientation::Project(cv::Mat &state, cv::Mat &projection, void *param)
{
	TrackerOrientation *tracker  = (TrackerOrientation *)param;
	int                 count    = projection.rows;
	cv::Mat             rot_mat  = cv::Mat(3, 1, CV_64F, &(state.at<double>(0 + 0)));
	double              zeros[3] = {0};
	cv::Mat             zero_tra = cv::Mat(3, 1, CV_64F, zeros);
	projection.reshape(2, 1);
	cv::projectPoints(tracker->_object_model,
	                  rot_mat,
	                  zero_tra,
	                  (tracker->_camera->calib_K),
	                  (tracker->_camera->calib_D),
	                  projection);
	projection.reshape(1, count);
}

void
TrackerOrientation::Reset()
{
	_pose.Reset();
	_pose.Mirror(false, true, true);
}

// Pose difference is stored...
double
TrackerOrientation::Track(cv::Mat &image)
{
	UpdateRotationOnly(image, image = cv::Mat());
	return 0;
}

bool
TrackerOrientation::UpdatePose(cv::Mat &image)
{
	int count_points = _F_v.size();
	if (count_points < 6)
		return false;

	cv::Mat _M                 = cv::Mat(count_points, 1, CV_64FC3);
	cv::Mat image_observations = cv::Mat(count_points * 2, 1, CV_64F); // [u v u v u v ...]'

	//map<int,Feature>::iterator it;
	int ind = 0;
	for (map<int, Feature>::iterator it = _F_v.begin(); it != _F_v.end(); it++) {
		if ((it->second.status3D == Feature::USE_FOR_POSE || it->second.status3D == Feature::IS_INITIAL)
		    && it->second.status2D == Feature::IS_TRACKED) {
			_M.at<double>(ind * 3 + 0) = it->second.point3d.x;
			_M.at<double>(ind * 3 + 1) = it->second.point3d.y;
			_M.at<double>(ind * 3 + 2) = it->second.point3d.z;

			image_observations.at<double>(ind * 2 + 0) = it->second.point.x;
			image_observations.at<double>(ind * 2 + 1) = it->second.point.y;
			ind++;
		}
	}

	if (ind < 6) {
		image_observations.release();
		_M.release();
		return false;
	}

	double  rot[3];
	cv::Mat rotm = cv::Mat(3, 1, CV_64F, rot);
	_pose.GetRodriques(rotm);

	cv::Mat par = cv::Mat(3, 1, CV_64F);
	memcpy(&(par.at<double>(0 + 0)), rot, 3 * sizeof(double));
	//par->data.db[3] = 0;

	cv::Rect r;
	r.x           = 0;
	r.y           = 0;
	r.height      = ind;
	r.width       = 1;
	cv::Mat Msub  = _M(r);
	_object_model = Msub;

	r.height                       = 2 * ind;
	cv::Mat image_observations_sub = image_observations(r);

	alvar::Optimization *opt = new alvar::Optimization(3, 2 * ind);

	double foo = opt->Optimize(
	  par, image_observations_sub, 0.0005, 5, Project, this, alvar::Optimization::TUKEY_LM);
	memcpy(rot, &(par.at<double>(0 + 0)), 3 * sizeof(double));
	_pose.SetRodriques(rotm);

	delete opt;

	par.release();
	image_observations.release();
	_M.release();

	return true;
}

bool
TrackerOrientation::UpdateRotationOnly(cv::Mat &gray, cv::Mat &image)
{
	if (gray.channels() != 1)
		return false;
	if (_grsc.empty())
		_grsc = cv::Mat(cv::Size(_xres, _yres), 8, 1);
	if ((_xres != _grsc.cols) || (_yres != _grsc.rows))
		cv::resize(gray, _grsc, _grsc.size());
	else
		_grsc.data = gray.data;

	map<int, Feature>::iterator it;
	for (it = _F_v.begin(); it != _F_v.end(); it++)
		it->second.status2D = Feature::NOT_TRACKED;

	// Track features in image domain (distorted points)
	_ft.Track(_grsc);

	// Go through image features and match to previous (_F_v)
	for (int i = 0; i < _ft.feature_count; i++) {
		int id         = _ft.ids[i];
		_F_v[id].point = _ft.features[i];
		_F_v[id].point.x *= _image_scale;
		_F_v[id].point.y *= _image_scale;
		_F_v[id].status2D = Feature::IS_TRACKED;

		// Throw outlier away
		if (_F_v[id].status3D == Feature::IS_OUTLIER) {
			_ft.DelFeature(i);
		}
	}

	// Delete features that are not tracked
	//		map<int,Feature>::iterator
	it = _F_v.begin();
	while (it != _F_v.end()) {
		if (it->second.status2D == Feature::NOT_TRACKED)
			_F_v.erase(it++);
		else
			++it;
	}

	// Update pose based on current information
	UpdatePose(image = cv::Mat());

	it = _F_v.begin();
	while (it != _F_v.end()) {
		Feature *f = &(it->second);

		// Add new points
		if (f->status3D == Feature::NONE) {
			double wx, wy, wz;

			cv::Point2f fpu = f->point;
			_camera->Undistort(fpu);

			// Tassa asetetaan z = inf ja lasketaan x ja y jotenkin?!?
			int object_scale = 1; // TODO Same as the pose?!?!?!?

			// inv(K)*[u v 1]'*scale
			wx = object_scale * (fpu.x - _camera->calib_K_data[0][2]) / _camera->calib_K_data[0][0];
			wy = object_scale * (fpu.y - _camera->calib_K_data[1][2]) / _camera->calib_K_data[1][1];
			wz = object_scale;

			// Now the points are in camera coordinate frame.
			alvar::Pose p = _pose;
			p.Invert();

			double  Xd[4] = {wx, wy, wz, 1};
			cv::Mat Xdm   = cv::Mat(4, 1, CV_64F, Xd);
			double  Pd[16];
			cv::Mat Pdm = cv::Mat(4, 4, CV_64F, Pd);
			p.GetMatrix(Pdm);
			Xdm          = Pdm * Xdm;
			f->point3d.x = Xd[0] / Xd[3];
			f->point3d.y = Xd[1] / Xd[3];
			f->point3d.z = Xd[2] / Xd[3];
			//cout<<f->point3d.z<<endl;

			f->status3D = Feature::USE_FOR_POSE;
		}

		if (!image.empty()) {
			if (f->status3D == Feature::NONE)
				cv::circle(image, cv::Point(int(f->point.x), int(f->point.y)), 3, CV_RGB(255, 0, 0), 1);
			else if (f->status3D == Feature::USE_FOR_POSE)
				cv::circle(image, cv::Point(int(f->point.x), int(f->point.y)), 3, CV_RGB(0, 255, 0), 1);
			else if (f->status3D == Feature::IS_INITIAL)
				cv::circle(image, cv::Point(int(f->point.x), int(f->point.y)), 3, CV_RGB(0, 0, 255), 1);
			else if (f->status3D == Feature::IS_OUTLIER)
				cv::circle(image, cv::Point(int(f->point.x), int(f->point.y)), 2, CV_RGB(255, 0, 255), 1);
		}

		// Delete points that bk error is too big
		// OK here just change state...
		// NYT TEHAAN TURHAAN ASKEN MUKAAN OTETUILLE..
		if (f->status3D == Feature::USE_FOR_POSE || f->status3D == Feature::IS_INITIAL) {
			double  p3d[3] = {f->point3d.x, f->point3d.y, f->point3d.z};
			cv::Mat p3dm   = cv::Mat(1, 1, CV_64FC3, p3d);
			double  p2d[2];
			cv::Mat p2dm = cv::Mat(2, 1, CV_64F, p2d);
			p2dm.reshape(2, 1);

			double gl_mat[16];
			_pose.GetMatrixGL(gl_mat);
			_camera->ProjectPoints(p3dm, gl_mat, p2dm);

			if (!image.empty())
				cv::line(image,
				         cv::Point(int(p2d[0]), int(p2d[1])),
				         cv::Point(int(f->point.x), int(f->point.y)),
				         CV_RGB(255, 0, 255));

			double dist = (p2d[0] - f->point.x) * (p2d[0] - f->point.x)
			              + (p2d[1] - f->point.y) * (p2d[1] - f->point.y);
			if (dist > _outlier_limit)
				f->status3D = Feature::IS_OUTLIER;
		}

		it++;
	}
	return true;
}

} // namespace alvar
