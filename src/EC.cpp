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

#include "EC.h"

#include "Optimization.h"

#include <opencv2/calib3d/calib3d_c.h>

namespace alvar {

struct ProjectParams
{
	Camera *      camera;
	const cv::Mat object_model;
};

static void
ProjectRot(cv::Mat &state, cv::Mat &projection, void *x)
{
	ProjectParams *foo          = (ProjectParams *)x;
	Camera *       camera       = foo->camera;
	const cv::Mat  object_model = foo->object_model;
	int            count        = projection->rows;
	cv::Mat        rot_mat      = cv::Mat(3, 1, CV_64F, &(state->data.db[0 + 0]));
	double         zeros[3]     = {0};
	cv::Mat        zero_tra     = cv::Mat(3, 1, CV_64F, zeros);
	cvReshape(projection, projection, 2, 1);
	cvProjectPoints2(
	  object_model, &rot_mat, &zero_tra, &(camera->calib_K), &(camera->calib_D), projection);
	cvReshape(projection, projection, 1, count);
}

// TODO: How this differs from the Camera::ProjectPoints ???
static void
Project(cv::Mat &state, cv::Mat &projection, void *x)
{
	ProjectParams *foo          = (ProjectParams *)x;
	Camera *       camera       = foo->camera;
	const cv::Mat  object_model = foo->object_model;
	int            count        = projection->rows;
	cv::Mat        rot_mat      = cv::Mat(3, 1, CV_64F, &(state->data.db[0 + 0]));
	cv::Mat        tra_mat      = cv::Mat(3, 1, CV_64F, &(state->data.db[0 + 3]));
	cvReshape(projection, projection, 2, 1);
	cvProjectPoints2(
	  object_model, &rot_mat, &tra_mat, &(camera->calib_K), &(camera->calib_D), projection);
	cvReshape(projection, projection, 1, count);
}

bool
CameraEC::UpdatePose(const cv::Mat &object_points,
                     cv::Mat &      image_points,
                     Pose *         pose,
                     cv::Mat &      weights)
{
	double  rot[3];
	cv::Mat rotm = cv::Mat(3, 1, CV_64F, rot);
	double  tra[3];
	cv::Mat tram = cv::Mat(3, 1, CV_64F, tra);
	pose->GetRodriques(&rotm);
	pose->GetTranslation(&tram);
	bool ret = UpdatePose(object_points, image_points, &rotm, &tram, weights);
	pose->SetRodriques(&rotm);
	pose->SetTranslation(&tram);
	return ret;
}

bool
CameraEC::UpdatePose(const cv::Mat &object_points,
                     cv::Mat &      image_points,
                     cv::Mat &      rot,
                     cv::Mat &      tra,
                     cv::Mat &      weights)
{
	if (object_points->height < 4)
		return false;
	/*	if (object_points->height < 6) {
		return false;
		// TODO: We need to change image_points into CV_32FC2
		return Camera::CalcExteriorOrientation(object_points, image_points, rot, tra);
	}*/
	cv::Mat par = cvCreateMat(6, 1, CV_64F);
	memcpy(&(par->data.db[0 + 0]), rot->data.db, 3 * sizeof(double));
	memcpy(&(par->data.db[0 + 3]), tra->data.db, 3 * sizeof(double));

	ProjectParams pparams;
	pparams.camera       = this;
	pparams.object_model = object_points;

	alvar::Optimization *opt = new alvar::Optimization(6, image_points->height);
	double               tmp = opt->Optimize(
    par, image_points, 0.0005, 2, Project, &pparams, alvar::Optimization::TUKEY_LM, 0, 0, weights);

	memcpy(rot->data.db, &(par->data.db[0 + 0]), 3 * sizeof(double));
	memcpy(tra->data.db, &(par->data.db[0 + 3]), 3 * sizeof(double));

	delete opt;

	cvReleaseMat(&par);
	return true;
}

bool
CameraEC::UpdateRotation(const cv::Mat &object_points, cv::Mat &image_points, Pose *pose)
{
	double  rot[3];
	cv::Mat rotm = cv::Mat(3, 1, CV_64F, rot);
	double  tra[3];
	cv::Mat tram = cv::Mat(3, 1, CV_64F, tra);
	pose->GetRodriques(&rotm);
	pose->GetTranslation(&tram);
	bool ret = UpdateRotation(object_points, image_points, &rotm, &tram);
	pose->SetRodriques(&rotm);
	pose->SetTranslation(&tram);
	return ret;
}

bool
CameraEC::UpdateRotation(const cv::Mat &object_points,
                         cv::Mat &      image_points,
                         cv::Mat &      rot,
                         cv::Mat &      tra)
{
	cv::Mat par = cvCreateMat(3, 1, CV_64F);
	memcpy(&(par->data.db[0 + 0]), rot->data.db, 3 * sizeof(double));
	ProjectParams pparams;
	pparams.camera           = this;
	pparams.object_model     = object_points;
	alvar::Optimization *opt = new alvar::Optimization(3, image_points->height);
	double               tmp = opt->Optimize(
    par, image_points, 0.0005, 2, ProjectRot, &pparams, alvar::Optimization::TUKEY_LM);
	memcpy(rot->data.db, &(par->data.db[0 + 0]), 3 * sizeof(double));
	delete opt;
	cvReleaseMat(&par);
	return true;
}

// Ol etta mirror asia on kunnossa
void
GetOrigo(Pose *pose, cv::Mat &O)
{
	pose->GetTranslation(O);
}

void
GetPointOnLine(const Pose *pose, Camera *camera, const cv::Point2f *u, cv::Mat &P)
{
	double  kid[9], rotd[9], trad[3], ud[3] = {u->x, u->y, 1};
	cv::Mat Ki = cv::Mat(3, 3, CV_64F, kid);
	cv::Mat R  = cv::Mat(3, 3, CV_64F, rotd);
	cv::Mat T  = cv::Mat(3, 1, CV_64F, trad);
	cv::Mat U  = cv::Mat(3, 1, CV_64F, ud);
	pose->GetMatrix(&R);
	pose->GetTranslation(&T);
	cvInv(&(camera->calib_K), &Ki);
	cv::MatMul(&R, &Ki, &Ki);
	cvGEMM(&Ki, &U, 1, &T, 1, P, 0);
}

bool
MidPointAlgorithm(cv::Mat &o1, cv::Mat &o2, cv::Mat &p1, cv::Mat &p2, cv::Point3f &X, double limit)
{
	double  ud[3], vd[3], wd[3];
	cv::Mat u = cv::Mat(3, 1, CV_64F, ud);
	cv::Mat v = cv::Mat(3, 1, CV_64F, vd);
	cv::Mat w = cv::Mat(3, 1, CV_64F, wd);

	cvSub(p1, o1, &u);
	cvSub(p2, o2, &v);
	cvSub(o1, o2, &w);

	double a = cvDotProduct(&u, &u);
	double b = cvDotProduct(&u, &v);
	double c = cvDotProduct(&v, &v);
	double d = cvDotProduct(&u, &w);
	double e = cvDotProduct(&v, &w);
	double D = a * c - b * b;
	double sc, tc;

	// compute the line parameters of the two closest points
	if (D < limit) {
		return false;
		// the lines are almost parallel
		sc = 0.0;
		tc = (b > c ? d / b : e / c); // use the largest denominator
	} else {
		sc = (b * e - c * d) / D;
		tc = (a * e - b * d) / D;
	}

	double  m1d[3], m2d[3];
	cv::Mat m1 = cv::Mat(3, 1, CV_64F, m1d);
	cv::Mat m2 = cv::Mat(3, 1, CV_64F, m2d);
	cvAddWeighted(&u, sc, o1, 1.0, 0.0, &m1);
	cvAddWeighted(&v, tc, o2, 1.0, 0.0, &m2);
	cvAddWeighted(&m1, 0.5, &m2, 0.5, 0.0, &m1);

	X.x = (float)m1d[0];
	X.y = (float)m1d[1];
	X.z = (float)m1d[2];

	return true;
}

// todo
bool
CameraEC::ReconstructFeature(const Pose *       pose1,
                             const Pose *       pose2,
                             const cv::Point2f *u1,
                             const cv::Point2f *u2,
                             cv::Point3f *      p3d,
                             double             limit)
{
	double  o1d[3], o2d[3], p1d[3], p2d[3];
	cv::Mat o1 = cv::Mat(3, 1, CV_64F, o1d);
	cv::Mat o2 = cv::Mat(3, 1, CV_64F, o2d);
	cv::Mat p1 = cv::Mat(3, 1, CV_64F, p1d);
	cv::Mat p2 = cv::Mat(3, 1, CV_64F, p2d);

	Pose po1 = *pose1; // Make copy so that we don't destroy the pose content
	Pose po2 = *pose2;
	po1.Invert();
	po2.Invert();
	GetOrigo(&po1, &o1);
	GetOrigo(&po2, &o2);
	GetPointOnLine(&po1, this, u1, &p1);
	GetPointOnLine(&po2, this, u2, &p2);

	return MidPointAlgorithm(&o1, &o2, &p1, &p2, *p3d, limit);
}

void
CameraEC::Get3dOnPlane(const Pose *pose, cv::Point2f p2d, cv::Point3f &p3d)
{
	double  pd[16], md[9], kd[9];
	cv::Mat P  = cv::Mat(4, 4, CV_64F, pd);
	cv::Mat H  = cv::Mat(3, 3, CV_64F, md);
	cv::Mat Ki = cv::Mat(3, 3, CV_64F, kd);

	pose->GetMatrix(&P);
	cvInv(&(calib_K), &Ki);

	// Construct homography from pose
	int ind_s = 0, ind_c = 0;
	for (int i = 0; i < 3; ++i) {
		CvRect r;
		r.x         = ind_s;
		r.y         = 0;
		r.height    = 3;
		r.width     = 1;
		cv::Mat sub = cv::Mat(3, 1, CV_64F);
		cvGetSubRect(&P, &sub, r);
		cv::Mat col = cv::Mat(3, 1, CV_64F);
		cvGetCol(&H, &col, ind_c);
		cvCopy(&sub, &col);
		ind_c++;
		ind_s++;
		if (i == 1)
			ind_s++;
	}

	// Apply H to get the 3D coordinates
	Camera::Undistort(p2d);
	double  xd[3] = {p2d.x, p2d.y, 1};
	cv::Mat X     = cv::Mat(3, 1, CV_64F, xd);
	cv::MatMul(&Ki, &X, &X);
	cvInv(&H, &H);
	cv::MatMul(&H, &X, &X);

	p3d.x = (float)(xd[0] / xd[2]);
	p3d.y = (float)(xd[1] / xd[2]);
	p3d.z = 0;
}

void
CameraEC::Get3dOnDepth(const Pose *pose, cv::Point2f p2d, float depth, cv::Point3f &p3d)
{
	double wx, wy, wz;
	Camera::Undistort(p2d);

	// Tassa asetetaan z = inf ja lasketaan x ja y jotenkin?!?
	//double object_scale = _dist_to_target; // TODO Same as the pose?!?!?!?

	// inv(K)*[u v 1]'*scale
	wx = depth * (p2d.x - calib_K_data[0][2]) / calib_K_data[0][0];
	wy = depth * (p2d.y - calib_K_data[1][2]) / calib_K_data[1][1];
	wz = depth;

	// Now the points are in camera coordinate frame.
	alvar::Pose p = *pose;
	p.Invert();

	double  Xd[4] = {wx, wy, wz, 1};
	cv::Mat Xdm   = cv::Mat(4, 1, CV_64F, Xd);
	double  Pd[16];
	cv::Mat Pdm = cv::Mat(4, 4, CV_64F, Pd);
	p.GetMatrix(&Pdm);
	cv::MatMul(&Pdm, &Xdm, &Xdm);
	p3d.x = float(Xd[0] / Xd[3]);
	p3d.y = float(Xd[1] / Xd[3]);
	p3d.z = float(Xd[2] / Xd[3]);
}

} // namespace alvar
