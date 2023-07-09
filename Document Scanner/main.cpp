#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/* Detect a document in the given image and warp it to a flat view */

// Globals
Mat imgOriginal, imgGray, imgBlur, imgCanny, imgThresh, imgDil, imgWarp, imgCrop;
vector<Point> initialPoints, docPoints;
float w = 420, h = 596;

// Detect edges and return dilated image
Mat preProc(Mat img) {

	cvtColor(img, imgGray, COLOR_BGR2GRAY);

	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);
	
	return imgDil;
}

vector<Point> getContours(Mat img) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPoly(contours.size()); // approximated polygon corners
	vector<Rect> boundRect(contours.size()); // bounding box corners

	vector<Point> biggest; // corner points of largest contour
	int maxArea = 0;

	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours[i]);

		// Filter out smaller contours
		if (area > 1000) {
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

			// Update maxArea only if contour is rectangle or square
			if (area > maxArea && conPoly[i].size() == 4) {
				maxArea = area;
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
			}
		}
	}

	return biggest;
}

// Reorders the corner points of the contour in: top left, top right, bot left, bot right
vector<Point> reorder(vector<Point> points) {

	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++){
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	int tl, tr, bl, br;
	tl = min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin();
	tr = max_element(subPoints.begin(), subPoints.end()) - subPoints.begin();
	bl = min_element(subPoints.begin(), subPoints.end()) - subPoints.begin();
	br = max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin();

	newPoints.push_back(points[tl]);
	newPoints.push_back(points[tr]);
	newPoints.push_back(points[bl]);
	newPoints.push_back(points[br]);

	return newPoints;
}

// Returns the warped document
Mat getWarp(Mat img, vector<Point> points, float w, float h) {

	Point2f src[4] = { points[0],points[1],points[2],points[3] };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat transfMatrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, transfMatrix, Size(w, h));

	return imgWarp;
}

int main() {

	string path = "Resources/paper.jpg";
	imgOriginal = imread(path);

	// Preprocessing
	imgThresh = preProc(imgOriginal);

	// Get biggest contour => detect document
	initialPoints = getContours(imgThresh);
	docPoints = reorder(initialPoints);

	// Warp the document
	imgWarp = getWarp(imgOriginal, docPoints, w, h);

	// Crop the warped document
	int cropval = 7;
	Rect roi(cropval, cropval, w - (2 * cropval), h - (2 * cropval));
	imgCrop = imgWarp(roi);

	imshow("Original Image", imgOriginal);
	imshow("Scanned Document", imgCrop);

	waitKey(0);

	return 0;
}