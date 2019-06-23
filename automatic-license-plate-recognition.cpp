#include "stdafx.h"
#include "common.h"
#include <queue>

using namespace cv;
using namespace std;

const char* WIN_SRC = "Source";
const char* WIN_DST = "Destination";
const char* WIN_BINARY = "Binary";
const char* WIN_COLOR = "Color";
const char* WIN_GRAYSCALE = "Grayscale";
const char* WIN_LABELS = "Labels";
const char* WIN_DETECTED = "Detected";
const char* WIN_RESULT = "Result";
const char* WIN_TABLITA = "Tablita";

const char* DETECTED = "_detected.bmp";
const char* LABELS = "_labels.bmp";
const char* BINARY = "_binary.bmp";
const char* GRAYSCALE = "_grayscale.bmp";
const char* TABLITA = "_tablita.bmp";

char source_name[MAX_PATH];
char detected_name[MAX_PATH];
char destination_name[MAX_PATH];

uchar neighbors[4];
int di[4] = { -1,0,1,0 };
int dj[4] = { 0,-1,0,1 };

int g_count_labels, i, j, x_1, x_2, y_1, y_2;

void set_name(char* source, char* destination, const char* extension);
void color_to_grayscale(Mat &source, Mat &destination);
void grayscale_to_binary(Mat &source, Mat &destination, int threshold);
void get_labels(Mat image, Mat &labels);
void open_image(char* source_name, char* destination_name);
void get_image(char* source_name, char* destination_name);
void get_text(char* source_name);
void show_result(char* source_name);
float euclidian_distance(Point2f p, Point2f q);

void main()
{
	while (openFileDlg(source_name))
	{
		open_image(source_name, detected_name);
		get_image(detected_name, destination_name);
		get_text(destination_name);
		show_result(source_name);
		waitKey();
	}
}

void open_image(char* source_name, char* destination_name)
{
	int height, width;
	float raport, AB, BC;
	vector<vector<Point>> contours;
	Mat color, grayscale, binary, canny_output;

	color = imread(source_name);
	resizeImg(color, color, 720, true);
	height = color.rows;
	width = color.cols;

	/* transformare din color */
	color_to_grayscale(color, grayscale);
	grayscale_to_binary(grayscale, binary, 100);

	Mat labels = Mat(height, width, CV_32SC1, Scalar(0));
	Mat labels_1 = Mat(height, width, CV_32SC1, Scalar(0));
	Mat detected = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

	get_labels(binary, labels);

	/* vector alocat dinamic cu dimensiunea egala cu numarul de etichete gasite */
	int *no_of_labels = (int*)calloc(g_count_labels + 1, sizeof(int));

	/* se calculeaza suprafata in pixeli pentru fiecare eticheta */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			no_of_labels[labels.at<int>(i, j)]++;
		}
	}

	/* daca suprafata unei etichete este mai mare decat suprafata fundalului care
	nu a primit o eticheta, atunci si acea eticheta e transformata in fundal */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (no_of_labels[labels.at<int>(i, j)] > no_of_labels[0])
			{
				labels.at<int>(i, j) = 0;
			}
		}
	}

	/* transforma fundalul fara eticheta in pixeli albi la imaginea binara */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (labels.at<int>(i, j) == 0)
			{
				binary.at<uchar>(i, j) = 255;
			}
		}
	}

	Mat clona = binary.clone();
	Canny(clona, canny_output, 100, 200);
	findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<RotatedRect> minRect(contours.size());
	for (i = 0; i < contours.size(); i++)
		minRect[i] = minAreaRect(contours[i]);

	Mat drawing = Mat(clona.size(), CV_8UC1, Scalar(0));
	Mat final_color = Mat(clona.size(), CV_8UC3, Scalar(0));
	Mat final_binary = Mat(clona.size(), CV_8UC1, Scalar(255));

	color.copyTo(final_color);

	/* caut toate patrulaterele care respecta proportiile tablitelor de inmatriculare */
	for (i = 0; i < contours.size(); i++)
	{
		Point2f rect_points[4];
		vector<Point> puncte;
		minRect[i].points(rect_points);

		if (rect_points[0].x <= rect_points[1].x && rect_points[1].x <= rect_points[2].x)
		{
			AB = euclidian_distance(rect_points[1], rect_points[0]);
			BC = euclidian_distance(rect_points[1], rect_points[2]);
		}
		else
		{
			AB = euclidian_distance(rect_points[3], rect_points[2]);
			BC = euclidian_distance(rect_points[1], rect_points[2]);
		}

		raport = AB / BC;

		if (raport > 3.5f && raport <= 5.5f)
		{
			puncte.push_back(rect_points[0]);
			puncte.push_back(rect_points[1]);
			puncte.push_back(rect_points[2]);
			puncte.push_back(rect_points[3]);
			fillConvexPoly(final_binary, puncte, 0);
		}
	}

	/* final_binary este o imagine cu dreptunghiuri
	imaginea va fi etichetata si se cauta dreptunghiul de dimensiune maxima */
	get_labels(final_binary, labels_1);

	int *no_of_labels_1 = (int*)calloc(g_count_labels + 1, sizeof(int));

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			no_of_labels_1[labels_1.at<int>(i, j)]++;
		}
	}

	/* caut dreptunghiul cu aria maxima */
	int area = 0;
	int index = 0;
	for (i = 1; i <= g_count_labels; i++)
	{
		if (no_of_labels_1[i] > area)
		{
			area = no_of_labels_1[i];
			index = i;
		}
	}

	Mat output = Mat(height, width, CV_8UC1, Scalar(255));

	/* output e imaginea binara formata doar din dreptunghiul cu aria maxima */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (labels_1.at<int>(i, j) == index)
			{
				output.at<uchar>(i, j) = 0;
			}
			else
			{
				output.at<uchar>(i, j) = 255;
			}
		}
	}
	
	/* detected primeste portiunea color din imaginea sursa
	unde output este masca pentru dreptunghiul cu placuta */
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (output.at<uchar>(i, j) == 0)
			{
				detected.at<Vec3b>(i, j) = color.at<Vec3b>(i, j);
			}
		}
	}

	/* salvare imagini */
	set_name(source_name, destination_name, DETECTED);
	imwrite(destination_name, detected);

	/* afisare imagini */
	imshow(WIN_SRC, color);
	imshow(WIN_DETECTED, detected);
}

void get_image(char* source_name, char* destination_name)
{
	Mat color, grayscale, binary;
	int height, width;
	int i, j;
	color = imread(source_name);
	height = color.rows;
	width = color.cols;

	//transformare din color
	color_to_grayscale(color, grayscale);
	grayscale_to_binary(grayscale, binary, 150);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (color.at<Vec3b>(i, j) != Vec3b(255, 255, 255))
			{
				x_1 = i;
				y_1 = j;
				goto here_1;
			}
		}
	}
here_1:
	for (i = height - 1; i >= 0; i--)
	{
		for (j = width - 1; j >= 0; j--)
		{
			if (color.at<Vec3b>(i, j) != Vec3b(255, 255, 255))
			{
				x_2 = i;
				y_2 = j;
				goto here_2;
			}
		}
	}
here_2:
	int len_x = x_2 - x_1;
	int len_y = y_2 - y_1;

	Mat output = Mat(len_x, len_y, CV_8UC1, Scalar(0));
	for (i = 0; i < len_x; i++)
	{
		for (j = 0; j < len_y; j++)
		{
			output.at<uchar>(i, j) = 255 - binary.at<uchar>(i + x_1, j + y_1);
		}
	}

	set_name(source_name, destination_name, TABLITA);
	imwrite(destination_name, output);

	imshow(WIN_TABLITA, output);
}

void get_text(char* source_name)
{
	Mat binary;
	int height, width, numb = 0;

	binary = imread(source_name, CV_LOAD_IMAGE_GRAYSCALE);
	height = binary.rows;
	width = binary.cols;

	Mat labels = Mat(height, width, CV_32SC1, Scalar(0));
	Mat output = Mat(height, width, CV_8UC1, Scalar(255));
	get_labels(binary, labels);

	int *no_of_labels = (int*)calloc(g_count_labels, sizeof(int));

	for (j = 0; j < width; j++)
	{
		if (labels.at<int>(height / 2, j) != 0)
			no_of_labels[labels.at<int>(height / 2, j)]++;
	}

	int labs = 0;
	for (j = 0; j < g_count_labels; j++)
	{
		if (no_of_labels[j] != 0)
		{
			labs++;
		}
	}

	int *vec = (int*)calloc(labs, sizeof(int));
	labs = 0;
	for (j = 0; j <= g_count_labels; j++)
	{
		if (no_of_labels[j] != 0)
		{
			vec[labs] = j;
			labs++;
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			for (int k = 0; k < labs; k++)
				if (labels.at<int>(i, j) == vec[k])
					output.at<uchar>(i, j) = 0;
		}
	}

	imshow(WIN_DST, output);
}

/* afiseaza imaginea originala cu placuta de inmatriculare incadrata intr-un dreptunghi */
void show_result(char* source_name)
{
	Mat matrix = imread(source_name);
	resizeImg(matrix, matrix, 720, true);
	rectangle(matrix, Point(y_1, x_1), Point(y_2, x_2), Scalar(0, 0, 255), 3, 8, 0);
	imshow(WIN_RESULT, matrix);
}

/* seteaza numele imaginii care urmeaza sa fie salvata */
void set_name(char* source, char* destination, const char* extension)
{
	int length = strlen(source);				//obtine lungimea path-ului
	while ('.' != *(source + length))			//analizeaza de la dreapta spre stanga codul pana trece de .(extensie)
	{
		length--;
	}
	strcpy(destination, source);				//copiez path-ul de la sursa in stringul name
	strcpy(destination + length, extension);	//fac overwrite cu extensia
}

/* transforma imaginea din color in grayscale */
void color_to_grayscale(Mat &source, Mat &destination)
{
	cvtColor(source, destination, CV_BGR2GRAY);
}

/* binarizeaza invers o imagine grayscale dupa un prag ales threshold */
void grayscale_to_binary(Mat &source, Mat &destination, int threshold)
{
	int height = source.rows;
	int width = source.cols;
	destination = source.clone();

	uchar *lpSrc = source.data;
	uchar *lpDst = destination.data;
	int w = (int)source.step;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = lpSrc[i*w + j];
			if (val < threshold)
			{
				lpDst[i*w + j] = 255;
			}
			else
			{
				lpDst[i*w + j] = 0;
			}
		}
	}
}

/* algoritm de etichetare - parcurgere in latime, laborator 5 */
void get_labels(Mat image, Mat &labels)
{
	int label = 1;
	int height = image.rows;
	int width = image.cols;
	std::queue<Point2i> Q;

	/* parcurge imaginea binara si scrie o matrice pentru etichete
    matricea de etichete rezultata dupa parcurgerea buclelor va fi o
	matrice cu toti pixelii negri, mai putin marginile de 3 pixeli albi */
	uchar *lpSrc = image.data;
	uchar *lpDst = labels.data;
	int w = (int)image.step;
	int x = (int)labels.step;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i < 3 || i > height - 3 || j < 3 || j > width - 3)
				image.at<uchar>(i, j) = 255;
			labels.at<int>(i, j) = 0;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (image.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
			{
				label++;
				labels.at<int>(i, j) = label;
				Point2i point = Point2i(i, j);
				Q.push(point);
				while (!Q.empty())
				{
					Point2i q = Q.front();
					int x = q.x;
					int y = q.y;
					Q.pop();
					for (int k = 0; k < 4; k++)
					{
						neighbors[k] = image.at<uchar>(x + di[k], y + dj[k]);
						if (image.at<uchar>(x + di[k], y + dj[k]) == 0 && labels.at<int>(x + di[k], y + dj[k]) == 0)
						{
							labels.at<int>(x + di[k], y + dj[k]) = label;
							Point2i neighbor = Point2i(x + di[k], y + dj[k]);
							Q.push(neighbor);
						}
					}
				}
			}
		}
	}
	g_count_labels = label;
}

/* returneaza distanta dintre doua puncte */
float euclidian_distance(Point2f p, Point2f q) {
	Point2f diff = p - q;
	return cv::sqrt(abs(diff.x*diff.x + diff.y*diff.y));
}
