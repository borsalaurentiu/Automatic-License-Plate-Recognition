// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>

#define ACUMULATOR 256
#define WH 5
#define TH 0.0003

//Lab3
int histogram[ACUMULATOR];
int histograma_cumulativa[ACUMULATOR];
float float_histogram[256];
float average_histogram[256];

//Lab5
#define HEIGHT 245
#define WIDTH 462
int di[8] = { -1,-1,-1,0,0,1,1,1 };
int dj[8] = { -1,0,1,-1,1,-1,0,1 };

int di9[9] = { -1,-1,-1,0,0,0,1,1,1 };
int dj9[9] = { -1,0,1,-1,0,1,-1,0,1 };

int d_i[4] = { -1,0,1,0 };
int d_j[4] = { 0,-1,0,1 };

int dir_i[8] = { 0,-1,-1,-1,0,1,1,1 };
int dir_j[8] = { 1,1,0,-1,-1,-1,0,1 };

uchar neighbors[8];
int zeros[HEIGHT*WIDTH];
//using namespace std;
//std::queue<Point2i> Q;



void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void negative_image() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

uchar schimbaGriAditiv(uchar factor_aditiv, uchar rgb) {
	if (0 == factor_aditiv)
		return rgb;

	if (factor_aditiv > 0)
	{
		if (255 - rgb > factor_aditiv)
			rgb += factor_aditiv;
		else
			return 255;
	}
	else
	{
		if (rgb > (0 - factor_aditiv))
			rgb += factor_aditiv;
		else
			return 0;
	}

	return rgb;
}

uchar schimbaGriMultiplicativ(uchar factor_multiplicativ, uchar rgb) {
	if (0 == factor_multiplicativ)
		return 1;

	if (factor_multiplicativ * rgb > 255)
		return 255;
	else
		rgb *= factor_multiplicativ;

	return rgb;
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testColor2GrayLab1Pb3()
{
	uchar factor_aditiv = 155;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = schimbaGriAditiv(factor_aditiv, (r + g + b) / 3);
			}
		}

		imshow("input image", src);
		imshow("gray image 3", dst);
		waitKey();
	}
}

void testColor2GrayLab1Pb4()
{
	uchar factor_multiplicativ = 3;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = schimbaGriMultiplicativ(factor_multiplicativ, (r + g + b) / 3);
			}
		}

		imshow("input image", src);
		imshow("gray image 4", dst);
		waitKey();
	}
}

void imagineColor256Lab1Pb5()
{
	char fname[MAX_PATH];
	Mat img = Mat(256, 256, CV_8UC3);
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			pixel[0] = 255;
			pixel[1] = 255;
			pixel[2] = 255;
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = 0; i < height / 2; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			pixel[0] = 0;
			pixel[1] = 0;
			pixel[2] = 255;
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = height / 2; i < height; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			pixel[0] = 0;
			pixel[1] = 255;
			pixel[2] = 0;
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = height / 2; i < height; i++)
	{
		for (int j = width / 2; j < width; j++)
		{
			Vec3b pixel = img.at<Vec3b>(i, j);
			pixel[0] = 0;
			pixel[1] = 255;
			pixel[2] = 255;
			img.at<Vec3b>(i, j) = pixel;
		}
	}

	imshow("image", img);
	waitKey();
}

void printMatrix33()
{
	float matrix[3][3] = {
		3.0f, 0.0f, 2.0f,
		2.0f, 0.0f, -2.0f,
		0.0f, 1.0f, 1.0f
	};

	float transpusa[3][3];
	float tr[9];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			transpusa[i][j] = matrix[j][i];

	tr[0] = transpusa[1][1] * transpusa[2][2] - transpusa[1][2] * transpusa[2][1];
	tr[1] = transpusa[1][0] * transpusa[2][2] - transpusa[1][2] * transpusa[2][0];
	tr[2] = transpusa[1][0] * transpusa[2][1] - transpusa[1][1] * transpusa[2][0];

	tr[3] = transpusa[0][1] * transpusa[2][2] - transpusa[0][2] * transpusa[2][1];
	tr[4] = transpusa[0][0] * transpusa[2][2] - transpusa[0][2] * transpusa[2][0];
	tr[5] = transpusa[0][0] * transpusa[2][1] - transpusa[0][1] * transpusa[2][0];

	tr[6] = transpusa[0][1] * transpusa[1][2] - transpusa[0][2] * transpusa[1][1];
	tr[7] = transpusa[0][0] * transpusa[1][2] - transpusa[0][2] * transpusa[1][0];
	tr[8] = transpusa[0][0] * transpusa[1][1] - transpusa[0][1] * transpusa[1][0];


	float det = matrix[0][0] * matrix[1][1] * matrix[2][2] +
		matrix[0][1] * matrix[1][2] * matrix[2][0] +
		matrix[0][2] * matrix[1][0] * matrix[2][1] -
		matrix[2][0] * matrix[1][1] * matrix[0][2] -
		matrix[2][1] * matrix[0][0] * matrix[1][2] -
		matrix[2][2] * matrix[1][0] * matrix[0][1];

	printf("Determinant = %f\nPrint matrix: \n", det);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			printf("%f   ", matrix[i][j]);
		printf("\n");
	}

	if (0.0f != det)
	{
		float inversa[3][3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				inversa[i][j] = 1 / det * pow(-1, i + j) * tr[j + i * 3];
			}
		}
		printf("Determinant = %f\nPrint inverse: \n", det);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				printf("%f   ", inversa[i][j]);
			printf("\n");
		}

	}
	else
	{
		printf("Determinant == 0\n");
	}
	system("pause");
}

void print_inverse_matrixLab1Pb6(Mat matrix)
{
	Mat inverse = matrix.inv();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			printf("%f ", inverse.at<float>(i, j));
		}
		printf("\n");
	}

	waitKey();
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
	Input:
	name - destination (output) window name
	hist - pointer to the vector containing the histogram values
	hist_cols - no. of bins (elements) in the histogram = histogram image width
	hist_height - height of the histogram image
	Call example:
	showHistogram ("MyHist", hist_dir, 255, 200);
	*/
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(0, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void showFloatHistogram(const std::string& name, float* hist, const int  hist_cols, const int hist_height)
{
	/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
	Input:
	name - destination (output) window name
	hist - pointer to the vector containing the histogram values
	hist_cols - no. of bins (elements) in the histogram = histogram image width
	hist_height - height of the histogram image
	Call example:
	showHistogram ("MyHist", hist_dir, 255, 200);
	*/
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	float max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point2f p1 = Point2f(x, baseline);
		Point2f p2 = Point2f(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(0, 0, 0)); // histogram bins colored in red
	}

	imshow(name, imgHist);
}

//Lab2
void Lab2Pb1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dstB = Mat(height, width, CV_8UC3);
		Mat dstG = Mat(height, width, CV_8UC3);
		Mat dstR = Mat(height, width, CV_8UC3);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				dstB.at<Vec3b>(i, j) = Vec3b(v3[0], 0, 0);
				dstG.at<Vec3b>(i, j) = Vec3b(0, v3[1], 0);
				dstR.at<Vec3b>(i, j) = Vec3b(0, 0, v3[2]);
			}
		}

		imshow("input image", src);
		imshow("R", dstR);
		imshow("B", dstB);
		imshow("G", dstG);
		waitKey();
	}
}

void Lab2Pb2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				dst.at<uchar>(i, j) = (v3[0] + v3[1] + v3[2]) / 3;
			}
		}

		imshow("RGB image", src);
		imshow("Grayscale image", dst);
		waitKey();
	}
}

void Lab2Pb3(int treshold)
{
	//	uchar treshold = (uchar)tres;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < treshold)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}

		imshow("Src image", src);
		imshow("Dst image", dst);
		waitKey();
	}
}

void Lab2Pb4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		float minim, maxim, c, value, hue, saturation;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				minim = min(min((float)v3[0] / 255.0f, (float)v3[1] / 255.0f), (float)v3[2] / 255.0f);
				maxim = max(max((float)v3[0] / 255.0f, (float)v3[1] / 255.0f), (float)v3[2] / 255.0f);
				c = maxim - minim;
				value = maxim;
				if (value != 0)
				{
					saturation = c / value;
				}
				else
				{
					saturation = 0.0f;
				}

				if (c != 0) {
					/*if (i == 155 && j == 55) {
						printf("%c, %c, %c, %c\n", (uchar)(maxim *255), v3[2], v3[1], v3[0]);
					}*/
					if (maxim == (float)v3[2] / 255.0f)
						hue = 60 * (v3[1] - v3[0]) / (c *255.0f);
					if (maxim == (float)v3[1] / 255.0f)
						hue = 120 + 60 * (v3[0] - v3[2]) / (c*255.0f);
					if (maxim == (float)v3[0] / 255.0f)
						hue = 240 + 60 * (v3[2] - v3[1]) / (c*255.0f);
				}
				else // grayscale
					hue = 0;
				if (hue < 0)
					hue = hue + 360;

				//printf("%f\n", hue);
				H.at<uchar>(i, j) = hue * 255 / 360;
				S.at<uchar>(i, j) = saturation * 255;
				V.at<uchar>(i, j) = value * 255;
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void isInsideLab2Pb5(int i, int j)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		if (i > 0 && j > 0 && i <= height && j <= width)
		{
			printf("True\n");
			system("pause");
		}
		else
		{
			printf("False\n");
			system("pause");
		}
	}
}

//Lab3
void showHistogramLab3Pb1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
			}
		}

		showHistogram("Histograma", histogram, width, height);
		imshow("Src image", src);
		waitKey();
	}
}

void showHistogramLab3Pb2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int m = height * width;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			float_histogram[i] = (float)histogram[i] / m;
			printf("%d, %f\n", histogram[i], float_histogram[i]);
		}


		showHistogram("HistogramaInt", histogram, width, height);
		showFloatHistogram("HistogramaFloat", float_histogram, width, height);
		imshow("Src image", src);
		waitKey();
	}
}

void showHistogramLab3Pb4()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int m = height * width;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val * ACUMULATOR / 256]++;
			}
		}


		showHistogram("Histograma", histogram, width, height);
		imshow("Src image", src);
		waitKey();
	}
}

void showHistogramLab3Pb5()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		int m = height * width;
		int length = 2 * WH + 1;
		float treshold = TH;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			float_histogram[i] = (float)histogram[i] / m;
		}

		for (int k = WH; k < 256 - WH; k++)
		{
			float v = 0.0f;
			float max_v = 0.0f;
			for (int i = k - 5; i <= k + 5; i++)
			{
				if (float_histogram[i] > max_v)
					max_v = float_histogram[i];
				v += float_histogram[i];
			}
			average_histogram[k] = v / (2 * WH + 1);
			if (float_histogram[k] > average_histogram[k] + treshold && float_histogram[k] >= max_v);
			printf("%f, %f\n", float_histogram[k], average_histogram[k]);

		}

		showHistogram("HistogramaInt", histogram, width, height);
		showFloatHistogram("HistogramaFloat", float_histogram, width, height);
		showFloatHistogram("HistogramaAverage", average_histogram, width, height);
		imshow("Src image", src);
		waitKey();
	}
}

void algoritm_1_Lab5Pb1()
{
	std::queue<Point2i> Q;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;
		int label = 1;
		Mat labels = Mat(height, width, CV_16SC1, zeros);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (img.at<uchar>(i, j) == 0 && labels.at<short>(i, j) == 0)
				{
					label++;
					labels.at<short>(i, j) = label;
					Point2i point = Point2i(i, j);
					Q.push(point);
					while (!Q.empty())
					{
						Point2i q = Q.front();
						int x = q.x;
						int y = q.y;
						Q.pop();
						for (int k = 0; k < 8; k++)
						{
							neighbors[k] = img.at<uchar>(x + di[k], y + dj[k]);
							if (img.at<uchar>(x + di[k], y + dj[k]) == 0 && labels.at<short>(x + di[k], y + dj[k]) == 0)
							{
								labels.at<short>(x + di[k], y + dj[k]) = label;
								Point2i neighbor = Point2i(x + di[k], y + dj[k]);
								Q.push(neighbor);
							}
						}
					}
				}
			}
		}
		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);
		uchar B, G, R;
		Vec3b pixel;
		Mat dst = Mat(height, width, CV_8UC3);

		while (label != 1)
		{
			B = d(gen);
			G = d(gen);
			R = d(gen);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (labels.at<short>(i, j) == label)
					{
						pixel = Vec3b(B, G, R);
						dst.at<Vec3b>(i, j) = pixel;
					}
				}
			}
			label--;
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<short>(i, j) == 0)
				{
					pixel = Vec3b(255, 255, 255);
					dst.at<Vec3b>(i, j) = pixel;
				}
			}
		}
		imshow("Source image", img);
		imshow("Label image", dst);
		waitKey();
	}
}

void algoritm_1_Lab5Pb3()
{
	std::queue<Point2i> Q;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = img.rows;
		int width = img.cols;
		int label = 1;
		Mat labels = Mat(height, width, CV_16SC1, zeros);
		std::vector<std::vector<int>> edges;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (img.at<uchar>(i, j) == 0 && labels.at<short>(i, j) == 0)
				{
					//L = vector() //aici
					//std::vector<int> L = ;
				}
			}
		}


		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);
		uchar B, G, R;
		Vec3b pixel;
		Mat dst = Mat(height, width, CV_8UC3);

		while (label != 1)
		{
			B = d(gen);
			G = d(gen);
			R = d(gen);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (labels.at<short>(i, j) == label)
					{
						pixel = Vec3b(B, G, R);
						dst.at<Vec3b>(i, j) = pixel;
					}
				}
			}
			label--;
		}
		imshow("Source image", img);
		imshow("Label image", dst);
		waitKey();
	}
}

//Lab4
void onMouse(int event, int x, int y, int flags, void* matrix)
{
	Mat *src_pointer = (Mat*)matrix;
	Mat src = *src_pointer;

	int height = src.rows;
	int width = src.cols;
	bool flag = false;
	int perimetru = 0, arie = 0;
	float thinness_ratio = 0.0;

	if (event == CV_EVENT_LBUTTONDBLCLK)
	{
		Vec3b pixel = src.at<Vec3b>(y, x);
		printf("pos(x, y): %d, %d --- color(RGB): %d, %d, %d\n", x, y, pixel[2], pixel[1], pixel[0]);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (pixel == src.at<Vec3b>(i, j))
				{
					arie++;
				}
			}
		}

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				if (src.at<Vec3b>(i, j) == pixel)
				{
					if (flag == false)
					{
						for (int ii = i - 1; ii <= i + 1; ii++)
						{
							for (int jj = j - 1; jj <= j + 1; jj++)
							{
								if (src.at<Vec3b>(ii, jj) != pixel)
								{
									flag = true;
								}
							}
						}
					}
					else
					{
						perimetru++;
						flag = false;
					}
				}
			}
		}

		perimetru *= CV_PI / 4;
		thinness_ratio = 4 * CV_PI * arie / (perimetru*perimetru);
		printf("Arie: %d\n", arie);
		printf("Perimetru: %d\n", perimetru);
		printf("Factor: %f\n", thinness_ratio);
	}
}

void Lab4() {
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_COLOR);

		namedWindow(fname, 1);

		setMouseCallback(fname, onMouse, &src);

		imshow(fname, src);

		waitKey(0);
	}

}

// Lab 6
typedef struct point {
	int i;
	int j;
	int dir;
} borderPoint;

void algoritm_1_Lab6Pb1()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		borderPoint P0, P1, P[10000];
		//	int n = 0;
		std::vector<borderPoint> vector_border_point;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (255 != src.at<uchar>(i, j))
				{
					P[0].i = i;
					P[0].j = j;
					P[0].dir = 7;
					vector_border_point.push_back(P[0]);
					goto here;
				}
			}
		}
	here:

		std::cout << P[0].i << " " << P[0].j << " " << P[0].dir << std::endl;

		int n = 1, m = 0;
		while (n < 1000)
		{
			for (int k = P[n - 1].dir; k < P[n - 1].dir + 8; k++)
			{
				if (src.at<uchar>(P[0].i, P[0].j) == src.at<uchar>(P[n - 1].i + dir_i[k % 8], P[n - 1].j + dir_j[k % 8]))
				{
					P[n].i = P[n - 1].i + dir_i[k % 8];
					P[n].j = P[n - 1].j + dir_j[k % 8];
					if (P[n - 1].dir % 2 == 0)
					{
						P[n].dir = (k % 8 + 7) % 8;
					}
					else
					{
						P[n].dir = (k % 8 + 6) % 8;
					}
					vector_border_point.push_back(P[n]);
					break;
				}
			}
			n++;
			//here2:
				//printf("a1\n");
				/*if (P[0].i == P[n].i && P[0].j == P[n].j)
				{
					m = n;
					n = 0;
				}
				else
					n++;*/
		}
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

		for (int i = 0; i < 1000; i++)
		{
			dst.at<uchar>(P[i].i, P[i].j) = 0;
			std::cout << P[i].i << " " << P[i].j << " " << P[i].dir << std::endl;
		}
		imshow("dst", dst);
		waitKey();
	}
}

//Lab 7
Mat dilatare_n(int n, Mat src)
{
	Mat dst;
	src.copyTo(dst);
	if (n > 0)
	{
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}
		n--;
		return dilatare_n(n, dst);
	}
	else
	{
		return dst;
		waitKey();
	}
}

Mat eroziune_n(int n, Mat src)
{
	Mat dst;
	src.copyTo(dst);
	if (n > 0)
	{
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (src.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							dst.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}
		n--;
		return eroziune_n(n, dst);
	}
	else
	{
		return dst;
		waitKey();
	}
}

Mat inchidere_n(int n, Mat src)
{
	Mat dst, aux;
	src.copyTo(dst);
	if (n > 0)
	{
		src.copyTo(aux);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						aux.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}

		aux.copyTo(dst);

		for (int i = 0; i < aux.rows; i++)
		{
			for (int j = 0; j < aux.cols; j++)
			{
				if (aux.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (aux.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							dst.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}



		n--;
		return inchidere_n(n, dst);
	}
	else
	{
		return dst;
		waitKey();
	}
}

Mat deschidere_n(int n, Mat src)
{
	Mat dst, aux;
	src.copyTo(dst);
	if (n > 0)
	{

		src.copyTo(aux);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (src.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							aux.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}

		aux.copyTo(dst);

		for (int i = 0; i < aux.rows; i++)
		{
			for (int j = 0; j < aux.cols; j++)
			{
				if (aux.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}


		n--;
		return deschidere_n(n, dst);
	}
	else
	{
		return dst;
		waitKey();
	}
}

Mat dilatare_4(Mat src)
{
	Mat dst;
	src.copyTo(dst);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				for (int k = 0; k < 4; k++)
				{
					dst.at<uchar>(i + d_i[k], j + d_j[k]) = 0;
				}
			}
		}
	}
	return dst;

}

void dilatare(int n)
{
	Mat src, dst;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		dst = dilatare_n(n, src);
		/*src.copyTo(dst);
		for (int i = 0; i < src.rows; i++)
		{
		for (int j = 0; j < src.cols; j++)
		{
		if (src.at<uchar>(i, j) == 0)
		{
		for (int k = 0; k < 8; k++)
		{
		dst.at<uchar>(i + di[k], j + dj[k]) = 0;
		}
		}
		}
		}*/
		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void eroziune(int n)
{
	Mat src, dst;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		dst = eroziune_n(n, src);
		/*src.copyTo(dst);
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (src.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							dst.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}*/
		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void inchidere(int n)
{
	Mat src, aux, dst;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		dst = inchidere_n(n, src);

		/*
		src.copyTo(aux);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						aux.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}

		aux.copyTo(dst);

		for (int i = 0; i < aux.rows; i++)
		{
			for (int j = 0; j < aux.cols; j++)
			{
				if (aux.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (aux.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							dst.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}*/

		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
}

void deschidere(int n)
{
	Mat src, aux, dst;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		dst = deschidere_n(n, src);

		/*src.copyTo(aux);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						if (src.at<uchar>(i + di[k], j + dj[k]) == 255)
						{
							aux.at<uchar>(i, j) = 255;
							break;
						}
					}
				}
			}
		}

		aux.copyTo(dst);

		for (int i = 0; i < aux.rows; i++)
		{
			for (int j = 0; j < aux.cols; j++)
			{
				if (aux.at<uchar>(i, j) == 0)
				{
					for (int k = 0; k < 8; k++)
					{
						dst.at<uchar>(i + di[k], j + dj[k]) = 0;
					}
				}
			}
		}*/

		imshow("src", src);
		imshow("dst", dst);
		waitKey();
	}
};

Mat complement(Mat src)
{
	Mat aux;
	src.copyTo(aux);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			aux.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}
	return aux;
}

Mat intersectie(Mat src1, Mat src2)
{
	Mat dst;
	src1.copyTo(dst);
	for (int i = 0; i < src1.rows; i++)
	{
		for (int j = 0; j < src1.cols; j++)
		{
			if (src1.at<uchar>(i, j) == src2.at <uchar>(i, j))
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}

void extrage_contur()
{
	Mat src, aux, dst, dest;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		src.copyTo(aux);
		dst = eroziune_n(3, src);
		dst.copyTo(dest);
		/*
		for (int i = 0; i < aux.rows; i++)
		{
			for (int j = 0; j < aux.cols; j++)
			{
				aux.at<uchar>(i, j) = 255 - aux.at<uchar>(i, j);
				if (aux.at<uchar>(i, j) == dst.at <uchar>(i, j))
				{
					dest.at<uchar>(i, j) = 0;
				}
				else
				{
					dest.at<uchar>(i, j) = 255;
				}

			}
		}*/
		aux = complement(src);
		dest = intersectie(aux, dst);

		imshow("src", aux);
		imshow("dst", dst);
		imshow("dest", dest);
		waitKey();
	}
}

void umple_regiune()
{
	Mat src, Xk, Xk1, Ac, dst;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Xk = Mat(src.rows, src.cols, CV_8UC1);
		Xk1 = Mat(src.rows, src.cols, CV_8UC1);
		Xk.setTo(cv::Scalar(255, 255, 255));
		Xk1.setTo(cv::Scalar(255, 255, 255));
		Ac = complement(src);

		imshow("Complement", Ac);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					Xk.at<uchar>(i, j) = 0;
					goto aici;
				}
			}
		}
	aici:

		Xk = dilatare_4(Xk);
		Xk1 = intersectie(Xk, Ac); //X1 = X0 dilatat

		//imshow("Xk", Xk);
		//imshow("Xk1", Xk1);

		int m = 50;
		while (m)
		{
			Xk1.copyTo(Xk);
			Xk1 = intersectie(Ac, dilatare_4(Xk));
			m--;
		}
		imshow("Xk", Xk);
		imshow("Xk1", Xk1);

		waitKey();

	}
}

// Lab8
#define L 255
int height, width;

void Lab8_Pb1()
{
	char fname[MAX_PATH];
	int I = 0, M = 0;
	double g = 0.0f, sigma = 0.0f, aux, sum = 0.0f;
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		height = src.rows;
		width = src.cols;
		M = height * width;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
				I += (int)val;
			}
		}

		g = I / M;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				aux = (src.at<uchar>(i, j) - g) * (src.at<uchar>(i, j) - g);
				sum += aux;
			}
		}

		histograma_cumulativa[0] = histogram[0];
		for (int i = 1; i < ACUMULATOR; i++)
		{
			histograma_cumulativa[i] = histogram[i] + histogram[i - 1];
		}

		sigma = sqrt(sum / M);
		printf("Valoarea medie a intensitatii: %f", g);
		printf("Valoarea deviatia standart: %f", sigma);
		showHistogram("Histograma", histogram, width, height);
		showHistogram("Histograma cumulativa", histograma_cumulativa, width, height);
		imshow("Src image", src);
		waitKey();

	}
}

double calculeaza_prag(int *histograma_8, double T, int Imin, int Imax)
{
	int N1 = 0, N2 = 0;
	int sum1 = 0, sum2 = 0;
	double T_final = 0.0f, uG1 = 0.0f, uG2 = 0.0f;

	for (int g = Imin; g < T; g++)
		N1 += histograma_8[g];
	for (int g = T + 1; g <= Imax; g++)
		N2 += histograma_8[g];

	for (int g = Imin; g < T; g++)
		sum1 += g*histograma_8[g];
	for (int g = T + 1; g <= Imax; g++)
		sum2 += g*histograma_8[g];

	uG1 = sum1 / N1;
	uG2 = sum2 / N2;
	printf("uG1 = %f, uG2 = %f\n", uG1, uG2);
	T_final = (uG1 + uG2) / 2;
	return T_final;
}

void Lab8_Pb2()
{
	char fname[MAX_PATH];
	int I = 0, M = 0;
	double T_init = 128.0f, T_aux, eroare = 0.1f;

	int Imin = 0, Imax = 0;
	while (openFileDlg(fname))
	{
		Mat src1 = imread(fname);
		height = src1.rows;
		width = src1.cols;
		Mat src = Mat(height, width, CV_8UC1);
		cvtColor(src1, src, CV_BGR2GRAY);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < 150)
					src.at<uchar>(i, j) = 0;
				else

					src.at<uchar>(i, j) = 255;
			}
		}



		Mat dst = Mat(height, width, CV_8UC1);
		M = height * width;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
				I += (int)val;
			}
		}

		Imin = histogram[0];
		Imax = histogram[0];

		for (int i = 0; i < ACUMULATOR; i++)
		{
			if (Imin > histogram[i])
				Imin = histogram[i];
			if (Imax < histogram[i])
				Imax = histogram[i];
		}

		T_aux = calculeaza_prag(histogram, T_init, Imin, Imax);
		while (abs(T_aux - T_init) > eroare)
		{
			T_init = T_aux;
			T_aux = calculeaza_prag(histogram, T_init, Imin, Imax);
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) <= T_aux)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}



		//showHistogram("Histograma", histogram, width, height);
		imshow("Src image", src);
		imshow("Dst image", dst);
		waitKey();
	}
}

void Lab8_Pb3()
{
	char fname[MAX_PATH];
	int M = 0, goutMIN, goutMAX;
	float gamma;

	scanf("%d", &goutMIN);
	scanf("%d", &goutMAX);
	scanf("%f", &gamma);

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		height = src.rows;
		width = src.cols;
		Mat dstNegativ = Mat(height, width, CV_8UC1);
		Mat dstContrast = Mat(height, width, CV_8UC1);
		Mat dstGamma = Mat(height, width, CV_8UC1);

		M = height * width;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				histogram[(int)val]++;
			}
		}

		//negativ
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				dstNegativ.at<uchar>(i, j) = L - val;
			}
		}

		//contrast
		int ginMIN = 255;
		int ginMAX = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (ginMIN > val)
				{
					ginMIN = val;
				}
				if (ginMAX < val)
				{
					ginMAX = val;
				}
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				dstContrast.at<uchar>(i, j) = goutMIN + (val - ginMIN)* (goutMAX - goutMIN) / (ginMAX - ginMIN);
			}
		}

		//gamma
		float x, putere, dest;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				x = (float)val / L;
				putere = pow(x, gamma);
				dest = L * putere;
				dstGamma.at<uchar>(i, j) = (uchar)dest;

			}
		}
		printf("%f ", gamma);

		showHistogram("Histograma", histogram, width, height);
		imshow("Src image", src);
		imshow("Negativ", dstNegativ);
		imshow("Contrast", dstContrast);
		imshow("Gamma", dstGamma);
		waitKey();
	}
}

// Lab 9

int filtru_medie[9] = { 1,1,1,1,1,1,1,1,1 };
int filtru_gauss[9] = { 1,2,1,2,4,2,1,2,1 };
int filtru_laplace_1[9] = { 0,-1,0,1,4,-1,0,-1,0 };
int filtru_laplace_2[9] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
int filtru_hp_1[9] = { 0,-1,0,-1,5,-1,0,-1,0 };
int filtru_hp_2[9] = { -1,-1,-1,-1,9,-1,-1,-1,-1 };

void trece_josLab9()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		height = src.rows;
		width = src.cols;
		Mat medie = Mat::zeros(height, width, CV_8UC1);
		Mat dst1 = Mat::zeros(height, width, CV_8UC1);
		Mat dst2 = Mat::zeros(height, width, CV_8UC1);
		Mat dst3 = Mat::zeros(height, width, CV_8UC1);
		Mat dst4 = Mat::zeros(height, width, CV_8UC1);
		Mat dst5 = Mat::zeros(height, width, CV_8UC1);
		Mat dst6 = Mat::zeros(height, width, CV_8UC1);

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++)
			{
				int dst_1 = 0, dst_2 = 0, dst_3 = 0, dst_4 = 0, dst_5 = 0, dst_6 = 0;
				for (int k = 0; k < 9; k++)
				{
					dst_1 += filtru_medie[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					dst_2 += filtru_gauss[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					dst_3 += filtru_laplace_1[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					dst_4 += filtru_laplace_2[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					dst_5 += filtru_hp_2[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					dst_6 += filtru_hp_2[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
				}
				printf("%d\n", dst_3);
				dst1.at<uchar>(i, j) = abs(dst_1) / 9;
				dst2.at<uchar>(i, j) = abs(dst_2) / 16;
				dst3.at<uchar>(i, j) = abs(dst_3);
				dst4.at<uchar>(i, j) = abs(dst_4);
				dst5.at<uchar>(i, j) = abs(dst_5);
				dst6.at<uchar>(i, j) = abs(dst_6) / 18;
			}
		/*{
		medie.at<uchar>(i, j) = ((int)src.at<uchar>(i - 1, j - 1) + src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j + 1) +
			src.at<uchar>(i, j - 1) + src.at<uchar>(i, j) + src.at<uchar>(i, j + 1) +
			src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1)) / 9;

		gauss.at<uchar>(i, j) = ((int)src.at<uchar>(i - 1, j - 1) + 2 * src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j + 1) +
			2 * src.at<uchar>(i, j - 1) + 4 * src.at<uchar>(i, j) + 2 * src.at<uchar>(i, j + 1) +
			src.at<uchar>(i + 1, j - 1) + 2 * src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1)) / 16;
	}*/

		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++)
			{
				int dst_3 = 0;
				for (int k = 0; k < 9; k++)
				{
					dst_3 += filtru_laplace_1[k] * dst1.at<uchar>(i + di9[k], j + dj9[k]);
				}
				dst3.at<uchar>(i, j) = abs(dst_3) / 16;
			}

		imshow("Src", src);
		//imshow("Dst_1", dst1);
		//imshow("Dst_2", dst2);
		imshow("Dst_3", dst3);/*
		imshow("Dst_4", dst4);
		imshow("Dst_5", dst5);
		*/imshow("Dst_6", dst6);
		waitKey();
	}
}


//Lab11

int prewitt_1[9] = { -1,0,1,-1,0,1,-1,0,1 };
int prewitt_2[9] = { 1,1,1,0,0,0,-1,-1,-1 };

int sobel_1[9] = { -1,0,1,-2,0,2,-1,0,1 };
int sobel_2[9] = { 1,2,1,0,0,0,-1,-2,-1 };

int roberts_1[4] = { 1,0,0,-1 };
int roberts_2[4] = { 0,-1,1,0 };


void gradient()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat dst, dstFx, dstFy, src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		src.copyTo(dstFx);
		src.copyTo(dstFy);
		src.copyTo(dst);
		height = src.rows;
		width = src.cols;
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int fx = 0, fy = 0;
				float grad;
				float faza;
				for (int k = 0; k < 9; k++)
				{
					fx += sobel_1[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
					fy += sobel_2[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
				}
				if (fx < 0)
				{
					fx = 0;
				}
				if (fx > 255)
				{
					fx = 255;
				}
				/*while (fx > 255)
				{
					fx %= 255;
				}*/
				if (fy < 0)
				{
					fy = 0;
				}
				if (fy > 255)
				{
					fx = 255;
				}
				/*while (fy > 255)
				{
					fy %= 255;
				}*/
				grad = sqrt(fx*fx+fy*fy);
				faza = atan(fy / fx);
				dst.at<uchar>(i, j) = (uchar)grad/(4*sqrt(2));
//				printf(grad);
				dstFx.at<uchar>(i, j) = (uchar)fx;
				dstFy.at<uchar>(i, j) = (uchar)fy;
			}
		}

		imshow("SRC", src);
		imshow("DST", dst);
		imshow("DSTx", dstFx);
		imshow("DSTy", dstFy);
		waitKey();
	}
}



void Roberts_cross()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat dst, dstFx, dstFy, src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		src.copyTo(dstFx);
		src.copyTo(dstFy);
		src.copyTo(dst);
		height = src.rows;
		width = src.cols;
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int fx = 0, fy = 0;
				float grad;
				float faza;
				for (int k = 0; k < 4; k++)
				{
					fx += sobel_1[k] * src.at<uchar>(i + d_i[k], j + dj9[k]);
					fy += sobel_2[k] * src.at<uchar>(i + di9[k], j + dj9[k]);
				}
				if (fx < 0)
				{
					fx = 0;
				}
				if (fx > 255)
				{
					fx = 255;
				}
				/*while (fx > 255)
				{
				fx %= 255;
				}*/
				if (fy < 0)
				{
					fy = 0;
				}
				if (fy > 255)
				{
					fx = 255;
				}
				/*while (fy > 255)
				{
				fy %= 255;
				}*/
				grad = sqrt(fx*fx + fy*fy);
				faza = atan(fy / fx);
				dst.at<uchar>(i, j) = (uchar)grad / (4 * sqrt(2));
				//				printf(grad);
				dstFx.at<uchar>(i, j) = (uchar)fx;
				dstFy.at<uchar>(i, j) = (uchar)fy;
			}
		}

		imshow("SRC", src);
		imshow("DST", dst);
		imshow("DSTx", dstFx);
		imshow("DSTy", dstFy);
		waitKey();
	}
}

int main()
{
	float vals[9] = {
		3.0f, 0.0f, 2.0f,
		2.0f, 0.0f, -2.0f,
		0.0f, 1.0f, 1.0f
	};
	Mat matrix = Mat(3, 3, CV_32FC1, vals);
	int op, n;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			gradient();
			//testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			testColor2Gray();
			//testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 12:
			negative_image();
			break;
		case 13:
			testColor2GrayLab1Pb3();
			break;
		case 14:
			testColor2GrayLab1Pb4();
			break;
		case 15:
			imagineColor256Lab1Pb5();
			break;
		case 16:
			//printMatrix33();
			print_inverse_matrixLab1Pb6(matrix);
			break;
		case 21:
			Lab2Pb1();
			break;
		case 22:
			Lab2Pb2();
			break;
		case 23:
			int treshold;
			scanf("%d", &treshold);
			Lab2Pb3(treshold);
			break;
		case 24:
			Lab2Pb4();
			break;
		case 25:
			isInsideLab2Pb5(-50, 40);
			break;
		case 31:
			showHistogramLab3Pb1();
			break;
		case 32:
			showHistogramLab3Pb2();
			break;
		case 33:
			break;
		case 34:
			showHistogramLab3Pb4();
			break;
		case 35:
			showHistogramLab3Pb5();
			break;
		case 40:
			Lab4();
			break;
		case 51:
			algoritm_1_Lab5Pb1();
			system("pause");
			break;
		case 61:
			algoritm_1_Lab6Pb1();
			system("pause");
			break;
		case 71:
			scanf("%d", &n);
			dilatare(n);
			break;
		case 72:
			scanf("%d", &n);
			eroziune(n);
			break;
		case 73:
			scanf("%d", &n);
			inchidere(n);
			break;
		case 74:
			scanf("%d", &n);
			deschidere(n);
			break;
		case 75:
			extrage_contur();
			break;
		case 76:
			umple_regiune();
			break;
		case 81:
			Lab8_Pb1();
			break;
		case 82:
			Lab8_Pb2();
			break;
		case 83:
			Lab8_Pb3();
			break;
		case 91:
			trece_josLab9();
			break;
		}
	} while (op != 0);
	return 0;
}