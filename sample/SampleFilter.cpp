#include "AlvarException.h"
#include "Filter.h"
#include "Kalman.h"
#include "Platform.h"

#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace alvar;
using namespace std;

const int res = 320;

void
filter_none(double x, double y, double *fx, double *fy)
{
	*fx = x;
	*fy = y;
}

void
filter_average(double x, double y, double *fx, double *fy)
{
	static FilterAverage ax(30);
	static FilterAverage ay(30);
	*fx = ax.next(x);
	*fy = ay.next(y);
}

void
filter_median(double x, double y, double *fx, double *fy)
{
	static FilterMedian ax(30);
	static FilterMedian ay(30);
	*fx = ax.next(x);
	*fy = ay.next(y);
}

void
filter_running_average(double x, double y, double *fx, double *fy)
{
	static FilterRunningAverage ax(0.03);
	static FilterRunningAverage ay(0.03);
	*fx = ax.next(x);
	*fy = ay.next(y);
}

void
filter_des(double x, double y, double *fx, double *fy)
{
	static FilterDoubleExponentialSmoothing ax(0.03, 0.01);
	static FilterDoubleExponentialSmoothing ay(0.03, 0.01);
	*fx = ax.next(x);
	*fy = ay.next(y);
}

void
filter_kalman(double x, double y, double *fx, double *fy)
{
	static bool         init = true;
	static KalmanSensor sensor(4, 2);
	static Kalman       kalman(4); // x, y, dx, dy
	if (init) {
		init = false;
		// H
		sensor.H                  = cv::Mat::zeros(sensor.H.size(), sensor.H.type());
		sensor.H.at<double>(0, 0) = 1;
		sensor.H.at<double>(1, 1) = 1;
		// R
		cv::setIdentity(sensor.R, 10);
		// F
		cv::setIdentity(kalman.F);
		kalman.F.at<double>(0, 2) = 1;
		kalman.F.at<double>(1, 3) = 1;
		// Q
		kalman.Q.at<double>(0, 0) = 0.0001;
		kalman.Q.at<double>(1, 1) = 0.0001;
		kalman.Q.at<double>(2, 2) = 0.000001;
		kalman.Q.at<double>(3, 3) = 0.000001;
		// P
		cv::setIdentity(kalman.P, 100);
	}
	sensor.z.at<double>(0, 0) = x;
	sensor.z.at<double>(1, 0) = y;
	kalman.predict_update(&sensor,
	                      (unsigned long)(cv::getTickCount() / cv::getTickFrequency() * 1000));

	*fx = kalman.x.at<double>(0, 0);
	*fy = kalman.x.at<double>(1, 0);
}

void
filter_array_average(double x, double y, double *fx, double *fy)
{
	static bool                       init = true;
	static FilterArray<FilterAverage> fa(2);
	if (init) {
		init = false;
		for (int i = 0; i < 2; i++) {
			fa[i].setWindowSize(30);
		}
	}
	*fx = fa[0].next(x);
	*fy = fa[1].next(y);
}

class KalmanSensorOwn : public KalmanSensorEkf
{
	virtual void
	h(const cv::Mat &x_pred, cv::Mat &_z_pred)
	{
		double x                 = x_pred.at<double>(0, 0);
		double y                 = x_pred.at<double>(1, 0);
		double dx                = x_pred.at<double>(2, 0);
		double dy                = x_pred.at<double>(3, 0);
		_z_pred.at<double>(0, 0) = x;
		_z_pred.at<double>(0, 0) = y;
	}

public:
	KalmanSensorOwn(int _n, int _m) : KalmanSensorEkf(_n, _m)
	{
	}
};

class KalmanOwn : public KalmanEkf
{
	virtual void
	f(const cv::Mat &_x, cv::Mat &_x_pred, double dt)
	{
		double x                 = _x.at<double>(0, 0);
		double y                 = _x.at<double>(1, 0);
		double dx                = _x.at<double>(2, 0);
		double dy                = _x.at<double>(3, 0);
		_x_pred.at<double>(0, 0) = x + dt * dx;
		_x_pred.at<double>(1, 0) = y + dt * dy;
		_x_pred.at<double>(2, 0) = dx;
		_x_pred.at<double>(3, 0) = dy;
	}

public:
	KalmanOwn(int _n) : KalmanEkf(_n)
	{
	}
};

void
filter_ekf(double x, double y, double *fx, double *fy)
{
	static bool            init = true;
	static KalmanSensorOwn sensor(4, 2);
	static KalmanOwn       kalman(4); // x, y, dx, dy
	if (init) {
		init = false;
		// R
		cv::setIdentity(sensor.R, 100);
		// Q
		kalman.Q.at<double>(0, 0) = 0.001;
		kalman.Q.at<double>(1, 1) = 0.001;
		kalman.Q.at<double>(2, 2) = 0.01;
		kalman.Q.at<double>(3, 3) = 0.01;
		// P
		cv::setIdentity(kalman.P, 100);
	}
	sensor.z.at<double>(0, 0) = x;
	sensor.z.at<double>(1, 0) = y;
	kalman.predict_update(&sensor,
	                      (unsigned long)(cv::getTickCount() / cv::getTickFrequency() * 1000));
	*fx = kalman.x.at<double>(0, 0);
	*fy = kalman.x.at<double>(1, 0);
}

//Make list of filters
const int nof_filters                                                      = 8;
void (*(filters[nof_filters]))(double x, double y, double *fx, double *fy) = {
  filter_none,
  filter_average,
  filter_median,
  filter_running_average,
  filter_des,
  filter_kalman,
  filter_ekf,
  filter_array_average,
};
char filter_names[nof_filters][64] = {"No filter - Press any key to change",
                                      "Average",
                                      "Median",
                                      "Running Average",
                                      "Double Exponential Smoothing",
                                      "Kalman",
                                      "Extended Kalman",
                                      "Array (average)"};

// Just generate some random data that can be used as sensor input
void
get_measurement(double *x, double *y)
{
	static double xx  = 0;
	static double yy  = 0;
	static double dxx = 0.3;
	static double dyy = 0.7;
	xx += dxx;
	yy += dyy;
	if ((xx > res) || (xx < 0))
		dxx = -dxx;
	if ((yy > res) || (yy < 0))
		dyy = -dyy;
	double rx = (rand() * 20.0 / RAND_MAX) - 10.0;
	double ry = (rand() * 20.0 / RAND_MAX) - 10.0;

	// Add some outliers
	if (fabs(rx * ry) > 50) {
		rx *= 5;
		ry *= 5;
	}

	*x = xx + rx;
	*y = yy + ry;
}

int
main(int argc, char *argv[])
{
	try {
		// Output usage message
		std::string filename(argv[0]);
		filename = filename.substr(filename.find_last_of('\\') + 1);
		std::cout << "SampleFilter" << std::endl;
		std::cout << "============" << std::endl;
		std::cout << std::endl;
		std::cout << "Description:" << std::endl;
		std::cout << "  This is an example of how to use the 'FilterAverage', 'FilterMedian',"
		          << std::endl;
		std::cout << "  'FilterRunningAverage', 'FilterDoubleExponentialSmoothing', 'Kalman'"
		          << std::endl;
		std::cout << "  'KalmanEkf' and 'FilterArray' filtering classes. First the example"
		          << std::endl;
		std::cout << "  shows unfiltered test data with outliers. The data is then filtered"
		          << std::endl;
		std::cout << "  using the various filters. Press any key to cycle through the filters."
		          << std::endl;
		std::cout << std::endl;
		std::cout << "Usage:" << std::endl;
		std::cout << "  " << filename << std::endl;
		std::cout << std::endl;
		std::cout << "Keyboard Shortcuts:" << std::endl;
		std::cout << "  any key: cycle through filters" << std::endl;
		std::cout << "  q: quit" << std::endl;
		std::cout << std::endl;

		// Processing loop
		cv::Mat img = cv::Mat(cv::Size(res, res), CV_8UC1, 3);
		cv::namedWindow("SampleFilter");
		for (int ii = 0; ii < nof_filters; ii++) {
			int               key = 0;
			double            x, y;
			double            fx, fy;
			vector<cv::Point> tail;
			while (1) {
				get_measurement(&x, &y);
				filters[ii](x, y, &fx, &fy);
				img = cv::Mat::zeros(img.size(), img.type());
				cv::putText(img, filter_names[ii], cv::Point(3, res - 10), 0, 0.5, CV_RGB(255, 255, 255));
				cv::circle(img, cv::Point(int(x), int(y)), 2, CV_RGB(0, 255, 255));
				cv::circle(img, cv::Point(int(x), int(y)), 3, CV_RGB(255, 255, 255));
				cv::Point fp;
				fp.x = int(fx);
				fp.y = int(fy);
				tail.push_back(fp);
				for (size_t iii = 0; iii < tail.size(); iii++) {
					cv::circle(img, tail[iii], 0, CV_RGB(255, 255, 0));
				}
				cv::circle(img, fp, 2, CV_RGB(255, 0, 255));
				cv::imshow("SampleFilter", img);
				key = cv::waitKey(10);
				if (key != -1) {
					break;
				}
			}
			if (key == 'q') {
				break;
			}
		}
		img.release();
		return 0;
	} catch (const std::exception &e) {
		std::cout << "Exception: " << e.what() << endl;
	} catch (...) {
		std::cout << "Exception: unknown" << std::endl;
	}
}
