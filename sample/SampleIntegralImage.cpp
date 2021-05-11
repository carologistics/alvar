#include "CvTestbed.h"
#include "IntegralImage.h"
#include "Shared.h"

#include <opencv2/opencv.hpp>

using namespace alvar;
using namespace std;

void
videocallback(cv::Mat &image)
{
	static cv::Mat          img_grad  = cv::Mat();
	static cv::Mat          img_gray  = cv::Mat();
	static cv::Mat          img_ver   = cv::Mat();
	static cv::Mat          img_hor   = cv::Mat();
	static cv::Mat          img_canny = cv::Mat();
	static IntegralImage    integ;
	static IntegralGradient grad;

	if (img_gray.empty()) {
		// Following image is toggled visible using key '0'
		img_grad = CvTestbed::Instance().CreateImageWithProto("Gradient", image);
		CvTestbed::Instance().ToggleImageVisible(0);
		img_gray = CvTestbed::Instance().CreateImageWithProto("Grayscale", image, 0, 1);
		img_ver  = CvTestbed::Instance().CreateImage("Vertical", cv::Size(1, image.rows), CV_8UC1, 1);
		img_hor  = CvTestbed::Instance().CreateImage("Horizontal", cv::Size(image.cols, 1), CV_8UC1, 1);
		img_canny = CvTestbed::Instance().CreateImageWithProto("Canny", image, 0, 1);
	}
	if (image.channels() > 1) {
		cv::cvtColor(image, img_gray, cv::COLOR_RGB2GRAY);
	} else {
		image.copyTo(img_gray);
	}

	// Show PerformanceTimer
	//PerformanceTimer timer;
	//timer.Start();

	// Update the integral images
	integ.Update(img_gray);
	grad.Update(img_gray);

	// Whole image projections
	integ.GetSubimage(cv::Rect(0, 0, image.cols, image.rows), img_ver);
	integ.GetSubimage(cv::Rect(0, 0, image.cols, image.rows), img_hor);
	for (int y = 1; y < image.rows; y++) {
		img_ver.at<int>(y - 1, 0);
		cv::line(image,
		         cv::Point(img_ver.at<int>(y - 1, 0), y - 1),
		         cv::Point(img_ver.at<int>(y, 0), y),
		         CV_RGB(255, 0, 0));
	}
	for (int x = 1; x < image.cols; x++) {
		cv::line(image,
		         cv::Point(x - 1, img_hor.at<int>(0, x - 1)),
		         cv::Point(x, img_hor.at<int>(0, x)),
		         CV_RGB(0, 255, 0));
	}

	// Gradients
	// Mark gradients for 4x4 sub-blocks
	/*
    cvZero(img_grad);
    cv::Rect r = {0,0,4,4};
    for (int y=0; y<image.rows/4; y++) {
        r.y = y*4;
        for (int x=0; x<image.cols/4; x++) {
            r.x = x*4;
            double dirx, diry;
            grad.GetAveGradient(r, &dirx, &diry);
            cv::line(img_grad, cv::Point(r.x+2,r.y+2), cv::Point(r.x+2+int(dirx),r.y+2+int(diry)), CV_RGB(255,0,0));
        }
    }
    */

	// Gradients on canny
	img_grad      = cv::Mat::zeros(img_grad.size(), img_grad.type());
	static int t1 = 64, t2 = 192;
	cv::createTrackbar("t1", "Gradient", &t1, 255, NULL);
	cv::createTrackbar("t2", "Gradient", &t2, 255, NULL);
	cv::Canny(img_gray, img_canny, t1, t2);
	cv::Rect r = {0, 0, 4, 4};
	for (r.y = 0; r.y < img_canny.rows - 4; r.y++) {
		for (r.x = 0; r.x < img_canny.cols - 4; r.x++) {
			if (img_canny.data[r.y * img_canny.step + r.x]) {
				double dirx, diry;
				grad.GetAveGradient(r, &dirx, &diry);
				cv::line(img_grad,
				         cv::Point(r.x + 2, r.y + 2),
				         cv::Point(r.x + 2 + int(dirx), r.y + 2 + int(diry)),
				         CV_RGB(0, 0, 255));
				cv::line(img_grad,
				         cv::Point(r.x + 2, r.y + 2),
				         cv::Point(r.x + 2 + int(-diry), r.y + 2 + int(+dirx)),
				         CV_RGB(255, 0, 0));
				cv::line(img_grad,
				         cv::Point(r.x + 2, r.y + 2),
				         cv::Point(r.x + 2 + int(+diry), r.y + 2 + int(-dirx)),
				         CV_RGB(255, 0, 0));
			}
		}
	}

	// Show PerformanceTimer
	//cout<<"Processing: "<<1.0 / timer.Stop()<<" fps"<<endl;
}

int
main(int argc, char *argv[])
{
	try {
		// Output usage message
		std::string filename(argv[0]);
		filename = filename.substr(filename.find_last_of('\\') + 1);
		std::cout << "SampleIntegralImage" << std::endl;
		std::cout << "===================" << std::endl;
		std::cout << std::endl;
		std::cout << "Description:" << std::endl;
		std::cout << "  This is an example of how to use the 'IntegralImage' and" << std::endl;
		std::cout << "  'IntegralGradient' classes. The vertical (green) and horizontal (red)"
		          << std::endl;
		std::cout << "  whole image projections are computed using 'IntegralImage::GetSubimage'"
		          << std::endl;
		std::cout << "  and shown in the SampleIntegralImage window. The gradients of the" << std::endl;
		std::cout << "  image edges are shown in the Gradient window. The edges are detected"
		          << std::endl;
		std::cout << "  using the Canny edge detector where t1 and t2 are parameters for the"
		          << std::endl;
		std::cout << "  Canny algorithm. The gradients are drawn in red and their local normals"
		          << std::endl;
		std::cout << "  are drawn in blue." << std::endl;
		std::cout << std::endl;
		std::cout << "Usage:" << std::endl;
		std::cout << "  " << filename << " [device]" << std::endl;
		std::cout << std::endl;
		std::cout << "    device    integer selecting device from enumeration list (default 0)"
		          << std::endl;
		std::cout << "              highgui capture devices are prefered" << std::endl;
		std::cout << std::endl;
		std::cout << "Keyboard Shortcuts:" << std::endl;
		std::cout << "  0: show/hide gradient image" << std::endl;
		std::cout << "  1: show/hide grayscale image" << std::endl;
		std::cout << "  2: show/hide vertical image" << std::endl;
		std::cout << "  3: show/hide horizontal image" << std::endl;
		std::cout << "  4: show/hide canny image" << std::endl;
		std::cout << "  q: quit" << std::endl;
		std::cout << std::endl;

		// Initialise CvTestbed
		CvTestbed::Instance().SetVideoCallback(videocallback);

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

			cap->start();
			cap->setResolution(640, 480);

			if (cap->loadSettings(settingsFilename.str())) {
				std::cout << "Loading settings: " << settingsFilename.str() << std::endl;
			}

			std::stringstream title;
			title << "SampleIntegralImage (" << cap->captureDevice().captureType() << ")";

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
