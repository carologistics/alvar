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

#include "IntegralImage.h"

namespace alvar {

void
IntIndex::update_next_step()
{
	next_step = step;
	estep += step_remainder;
	if (estep >= steps) {
		estep -= steps;
		next_step++;
	}
}

IntIndex::IntIndex(int _res, int _steps)
{
	res     = _res;
	steps   = _steps;
	operator=(0);
}
int
IntIndex::operator=(int v)
{
	index          = 0;
	step           = res / steps;
	step_remainder = res % steps;
	estep          = 0;
	update_next_step();
	while ((index + next_step - 1) < v)
		next();
	return index;
}
int
IntIndex::next()
{
	index += next_step;
	update_next_step();
	return index;
}
int
IntIndex::get() const
{
	return index;
}
int
IntIndex::get_next_step() const
{
	return next_step;
}
int
IntIndex::end() const
{
	return res;
}

IntegralImage::IntegralImage()
{
	sum = cv::Mat();
}
IntegralImage::~IntegralImage()
{
	sum.release();
}
void
IntegralImage::Update(cv::Mat &gray)
{
	if ((sum.empty()) || (sum.rows != gray.cols + 1) || (sum.cols != gray.rows + 1)) {
		sum.release();
		// TODO: Now we assume 'double' - is it ok?
		sum = cv::Mat(cv::Size(gray.cols + 1, gray.rows + 1), CV_64F, 1);
	}
	cv::integral(gray, sum);
}
double
IntegralImage::GetSum(cv::Rect &rect, int *count /*=0*/)
{
	int x1 = rect.x;
	int x2 = rect.x + rect.width; // Note, not -1
	int y1 = rect.y;
	int y2 = rect.y + rect.height;
	//cout<<x1<<","<<y1<<"-"<<x2<<","<<y2<<endl;
	/*
	double v = +cvGet2D(sum, y2, x2).val[0]
	           -cvGet2D(sum, y2, x1).val[0]
			   -cvGet2D(sum, y1, x2).val[0]
			   +cvGet2D(sum, y1, x1).val[0];
    */
	double v =
	  +((double *)sum.data)[y2 * sum.cols
	                        + x2] //this is a guess, data might be the closest to imageData -Sebastian
	  - ((double *)sum.data)[y2 * sum.cols + x1] - ((double *)sum.data)[y1 * sum.cols + x2]
	  + ((double *)sum.data)[y1 * sum.cols + x1];

	if (count)
		*count = rect.width * rect.height;
	return v;
}
double
IntegralImage::GetAve(cv::Rect &rect)
{
	int count = 1;
	return GetSum(rect, &count) / count;
}
void
IntegralImage::GetSubimage(const cv::Rect &rect, cv::Mat &sub)
{
	int yi = 0;
	for (IntIndex yy(rect.height, sub.rows); yy.get() != yy.end(); yy.next(), yi++) {
		int xi = 0;
		for (IntIndex xx(rect.width, sub.cols); xx.get() != xx.end(); xx.next(), xi++) {
			//cout<<"res: "<<sum.rows<<","<<sum.cols<<" - ";
			//cout<<xi<<","<<yi<<": "<<rect.x<<","<<rect.y<<": "<<xx.get()<<","<<yy.get()<<endl;
			cv::Rect r   = {rect.x + xx.get(), rect.y + yy.get(), xx.get_next_step(), yy.get_next_step()};
			double   ave = GetAve(r);

			//cvSet2D(sub, yi, xi, cvScalar(ave));
			// TODO: Now we assume 8-bit gray
			sub.data[yi * sub.step + xi] = (char)ave;
		}
	}
}
void
IntegralGradient::CalculatePointNormals(cv::Mat &gray)
{
	int width  = gray.cols - 1;
	int height = gray.rows - 1;
	if ((normalx.empty()) || (normalx.cols != width) || (normalx.rows != height)) {
		normalx.release();
		normaly.release();
		normalx = cv::Mat(cv::Size(width, height), CV_64F, 1);
		normaly = cv::Mat(cv::Size(width, height), CV_64F, 1);
	}
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			/*
            // As we assume top-left coordinates we have these reverse compared to Donahue1992
            double a4 = cvGet2D(gray, j, i+1).val[0];
            double a3 = cvGet2D(gray, j, i).val[0];
            double a2 = cvGet2D(gray, j+1, i).val[0];
            double a1 = cvGet2D(gray, j+1, i+1).val[0];
            // Normal vectors;
            double nx = (-a1+a2+a3-a4)/4; 
            double ny = (-a1-a2+a3+a4)/4;
            cvSet2D(normalx, j, i, cvScalar(nx));
            cvSet2D(normaly, j, i, cvScalar(ny));
			*/
			// As we assume top-left coordinates we have these reverse compared to Donahue1992
			// TODO: Now we assume 8-bit gray
			double a4 = (unsigned char)gray.data[(j)*gray.step + (i + 1)];
			double a3 = (unsigned char)gray.data[(j)*gray.step + (i)];
			double a2 = (unsigned char)gray.data[(j + 1) * gray.step + (i)];
			double a1 = (unsigned char)gray.data[(j + 1) * gray.step + (i + 1)];
			// Normal vectors;
			double nx                                      = (-a1 + a2 + a3 - a4) / 4;
			double ny                                      = (-a1 - a2 + a3 + a4) / 4;
			((double *)normalx.data)[j * normalx.cols + i] = nx;
			((double *)normaly.data)[j * normaly.cols + i] = ny;
		}
	}
}
IntegralGradient::IntegralGradient()
{
	normalx = 0;
	normaly = 0;
}
IntegralGradient::~IntegralGradient()
{
	normalx.release();
	normaly.release();
}
void
IntegralGradient::Update(cv::Mat &gray)
{
	CalculatePointNormals(gray);
	integx.Update(normalx);
	integy.Update(normaly);
}
void
IntegralGradient::GetGradient(cv::Rect &rect, double *dirx, double *diry, int *count /*=0*/)
{
	cv::Rect r = {rect.x, rect.y, rect.width - 1, rect.height - 1};
	if (count)
		*dirx = integx.GetSum(r, count);
	else
		*dirx = integx.GetSum(r);
	*diry = integy.GetSum(r);
}
void
IntegralGradient::GetAveGradient(cv::Rect &rect, double *dirx, double *diry)
{
	int count = 1;
	GetGradient(rect, dirx, diry, &count);
	*dirx /= count;
	*diry /= count;
}

} // namespace alvar
