#include "Camera.h"
#include "CvTestbed.h"
#include "Shared.h"
#include "TrackerFeatures.h"
#include "TrackerPsa.h"
#include "TrackerStat.h"

#include <opencv2/opencv.hpp>

using namespace alvar;
using namespace std;

int               tracker = 0;
bool              reset   = true;
CvFont            font;
std::stringstream calibrationFilename;

void
track_none(cv::Mat &image, cv::Mat &img_gray)
{
}

void
track_psa(cv::Mat &image, cv::Mat &img_gray)
{
	static TrackerPsa tracker_psa;
	static double     x, y;
	if (reset) {
		reset = false;
		x     = img_gray.cols / 2;
		y     = img_gray.rows / 2;
		tracker_psa.Track(img_gray); // To reset tracker call it twice
	}
	tracker_psa.Track(img_gray);
	tracker_psa.Compensate(&x, &y);
	cv::circle(image, cv::Point(int(x), int(y)), 10, cvScalar(CV_RGB(255, 0, 0)));
}

void
track_psa_rot(cv::Mat &image, cv::Mat &img_gray)
{
	static TrackerPsaRot tracker_psa_rot;
	static double        x, y, r;
	if (reset) {
		reset = false;
		x     = img_gray.cols / 2;
		y     = img_gray.rows / 2;
		r     = 0;
		tracker_psa_rot.Track(img_gray); // To reset tracker call it twice
	}
	tracker_psa_rot.Track(img_gray);
	tracker_psa_rot.Compensate(&x, &y);
	r += tracker_psa_rot.rotd;
	cv::circle(image, cv::Point(int(x), int(y)), 15, cvScalar(CV_RGB(255, 0, 0)));
	double r_rad = r * 3.1415926535 / 180;
	cv::line(image,
	         cv::Point(int(x), int(y)),
	         cv::Point(int(x - sin(r_rad) * 15), int(y + cos(r_rad) * 15)),
	         cvScalar(CV_RGB(255, 0, 0)));
}

void
track_stat(cv::Mat &image, cv::Mat &img_gray)
{
	static TrackerStat tracker_stat;
	static double      x, y;
	if (reset) {
		reset = false;
		x     = img_gray.cols / 2;
		y     = img_gray.rows / 2;
		tracker_stat.Track(img_gray); // To reset tracker call it twice
	}
	tracker_stat.Track(img_gray);
	tracker_stat.Compensate(&x, &y);
	cv::circle(image, cv::Point(int(x), int(y)), 10, cvScalar(CV_RGB(0, 255, 0)));
}

void
track_stat_rot(cv::Mat &image, cv::Mat &img_gray)
{
	static TrackerStatRot tracker_stat_rot;
	static double         x, y, r;
	if (reset) {
		reset = false;
		x     = img_gray.cols / 2;
		y     = img_gray.rows / 2;
		r     = 0;
		tracker_stat_rot.Track(img_gray); // To reset tracker call it twice
	}
	tracker_stat_rot.Track(img_gray);
	tracker_stat_rot.Compensate(&x, &y);
	r += tracker_stat_rot.rotd;
	cv::circle(image, cv::Point(int(x), int(y)), 15, cvScalar(CV_RGB(0, 255, 0)));
	double r_rad = r * 3.1415926535 / 180;
	cv::line(image,
	         cv::Point(int(x), int(y)),
	         cv::Point(int(x - sin(r_rad) * 15), int(y + cos(r_rad) * 15)),
	         cvScalar(CV_RGB(0, 255, 0)));
}

void
track_features(cv::Mat &image, cv::Mat &img_gray)
{
	static TrackerFeatures tracker_features(200, 190, 0.01, 0, 4, 6);
	if (reset) {
		reset = false;
		tracker_features.Reset();
	}
	tracker_features.Purge();
	tracker_features.Track(img_gray);
	for (int i = 0; i < tracker_features.feature_count; i++) {
		cv::circle(image,
		           cv::Point(int(tracker_features.features[i].x), int(tracker_features.features[i].y)),
		           2,
		           cvScalar(CV_RGB(tracker_features.ids[i] % 255,
		                           (tracker_features.ids[i] * 7) % 255,
		                           (tracker_features.ids[i] * 11) % 255)));
	}
}

const int nof_trackers                                              = 6;
void (*(trackers[nof_trackers]))(cv::Mat &image, cv::Mat &img_gray) = {
  track_none,
  track_psa,
  track_psa_rot,
  track_stat,
  track_stat_rot,
  track_features,
};
char tracker_names[nof_trackers][64] = {
  "No tracker - Press any key to change",
  "TrackerPsa",
  "TrackerPsaRot",
  "TrackerStat",
  "TrackerStatRot",
  "TrackerFeatures",
};

void
videocallback(cv::Mat &image)
{
	assert(image);
	static Camera  cam;
	static cv::Mat img_gray = NULL;
	static bool    init     = true;

	if (init) {
		init     = false;
		img_gray = CvTestbed::Instance().CreateImageWithProto("img_gray", image, 0, 1);
		cout << "Loading calibration: " << calibrationFilename.str();
		if (cam.SetCalib(calibrationFilename.str().c_str(), image.cols, image.rows)) {
			cout << " [Ok]" << endl;
		} else {
			cam.SetRes(image.cols, image.rows);
			cout << " [Fail]" << endl;
		}
	}
	if (image->nChannels == 1)
		cvCopy(image, img_gray);
	else
		cvCvtColor(image, img_gray, CV_RGB2GRAY);

	trackers[tracker](image, img_gray);
	cvPutText(image,
	          tracker_names[tracker],
	          cv::Point(3, image.rows - 20),
	          &font,
	          cvScalar(CV_RGB(255, 255, 255)));
}

int
keycallback(int key)
{
	if ((key == 'r') || (key == 't')) {
		reset = true;
		return 0;
	} else if ((key == 'n') || (key == ' ')) {
		tracker = ((tracker + 1) % nof_trackers);
		reset   = true;
		return 0;
	}
	return key;
}

int
main(int argc, char *argv[])
{
	try {
		// Output usage message
		std::string filename(argv[0]);
		filename = filename.substr(filename.find_last_of('\\') + 1);
		std::cout << "SampleTrack" << std::endl;
		std::cout << "===========" << std::endl;
		std::cout << std::endl;
		std::cout << "Description:" << std::endl;
		std::cout << "  This is an example of how to use the 'TrackerPsa', 'TrackerPsaRot',"
		          << std::endl;
		std::cout << "  'TrackerFeatures', 'TrackerStat' and 'TrackerStatRot' classes to" << std::endl;
		std::cout << "  track the optical flow of the video." << std::endl;
		std::cout << std::endl;
		std::cout << "Usage:" << std::endl;
		std::cout << "  " << filename << " [device]" << std::endl;
		std::cout << std::endl;
		std::cout << "    device    integer selecting device from enumeration list (default 0)"
		          << std::endl;
		std::cout << "              highgui capture devices are prefered" << std::endl;
		std::cout << std::endl;
		std::cout << "Keyboard Shortcuts:" << std::endl;
		std::cout << "  r,t: reset tracker" << std::endl;
		std::cout << "  n,space: cycle through tracking algorithms" << std::endl;
		std::cout << "  q: quit" << std::endl;
		std::cout << std::endl;

		// Initialize font
		cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0);

		// Initialise CvTestbed
		CvTestbed::Instance().SetVideoCallback(videocallback);
		CvTestbed::Instance().SetKeyCallback(keycallback);

		// Enumerate possible capture plugins
		CaptureFactory::CapturePluginVector plugins = CaptureFactory::instance()->enumeratePlugins();
		if (plugins.size() < 1) {
			std::cout << "Could not find any capture plugins." << std::endl;
			return 0;
		}

		// Display capture plugins
		std::cout << "Available Plugins: ";
		outputEnumeratedPlugins(plugins);
		std::cout << std::endl;

		// Enumerate possible capture devices
		CaptureFactory::CaptureDeviceVector devices = CaptureFactory::instance()->enumerateDevices();
		if (devices.size() < 1) {
			std::cout << "Could not find any capture devices." << std::endl;
			return 0;
		}

		// Check command line argument for which device to use
		int selectedDevice = defaultDevice(devices);
		if (argc > 1) {
			selectedDevice = atoi(argv[1]);
		}
		if (selectedDevice >= (int)devices.size()) {
			selectedDevice = defaultDevice(devices);
		}

		// Display capture devices
		std::cout << "Enumerated Capture Devices:" << std::endl;
		outputEnumeratedDevices(devices, selectedDevice);
		std::cout << std::endl;

		// Create capture object from camera
		Capture *   cap        = CaptureFactory::instance()->createCapture(devices[selectedDevice]);
		std::string uniqueName = devices[selectedDevice].uniqueName();

		// Handle capture lifecycle and start video capture
		// Note that loadSettings/saveSettings are not supported by all plugins
		if (cap) {
			std::stringstream settingsFilename;
			settingsFilename << "camera_settings_" << uniqueName << ".xml";
			calibrationFilename << "camera_calibration_" << uniqueName << ".xml";

			cap->start();
			cap->setResolution(640, 480);

			if (cap->loadSettings(settingsFilename.str())) {
				std::cout << "Loading settings: " << settingsFilename.str() << std::endl;
			}

			std::stringstream title;
			title << "SampleTrack (" << cap->captureDevice().captureType() << ")";

			CvTestbed::Instance().StartVideo(cap, title.str().c_str());

			if (cap->saveSettings(settingsFilename.str())) {
				std::cout << "Saving settings: " << settingsFilename.str() << std::endl;
			}

			cap->stop();
			delete cap;
		} else if (CvTestbed::Instance().StartVideo(0, argv[0])) {
		} else {
			std::cout << "Could not initialize the selected capture backend." << std::endl;
		}

		return 0;
	} catch (const std::exception &e) {
		std::cout << "Exception: " << e.what() << endl;
	} catch (...) {
		std::cout << "Exception: unknown" << std::endl;
	}
}
