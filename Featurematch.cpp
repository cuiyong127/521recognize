#if 0
#include <iostream>  
#include <fstream>  
#include <sstream>  

#include "opencv2/opencv.hpp"  

using namespace cv;
using namespace std;

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts);

bool refineMatchesWithHomography(
	const std::vector<cv::KeyPoint>& queryKeypoints,
	const std::vector<cv::KeyPoint>& trainKeypoints,
	float reprojectionThreshold, std::vector<cv::DMatch>& matches,
	cv::Mat& homography);

/** @function main */
int main(int argc, char* argv[]) {

	/************************************************************************/
	/* 特征点检测，特征提取，特征匹配，计算投影变换                            */
	/************************************************************************/

	// 读取图片  
	Mat img1Ori = imread("1.jpg", 0);
	Mat img2Ori = imread("2.jpg", 0);


	Mat tempimg1, tempimg2;
	resize(img1Ori, tempimg1, Size(128, 90));
	resize(img2Ori, tempimg2, Size(128, 90));
	// 缩小尺度  
	Mat img1, img2;
	resize(tempimg1, img1, Size(tempimg1.cols / 4, tempimg1.cols / 4));
	resize(tempimg2, img2, Size(tempimg2.cols / 4, tempimg2.cols / 4));

	cv::Ptr<cv::FeatureDetector> detector = new cv::ORB(1000);                        // 创建orb特征点检测  
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::FREAK(true, true);           // 用Freak特征来描述特征点  
	cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BFMatcher(cv::NORM_HAMMING,  // 特征匹配，计算Hamming距离  
		true);

	vector<KeyPoint> keypoints1;  // 用于保存图中的特征点     
	vector<KeyPoint> keypoints2;
	Mat descriptors1;               // 用于保存图中的特征点的特征描述  
	Mat descriptors2;

	detector->detect(img1, keypoints1);      // 检测第一张图中的特征点  
	detector->detect(img2, keypoints2);

	extractor->compute(img1, keypoints1, descriptors1);      // 计算图中特征点位置的特征描述  
	extractor->compute(img2, keypoints2, descriptors2);

	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches);

	Mat imResultOri;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, imResultOri,
		CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
	cout << "[Info] # of matches : " << matches.size() << endl;

	Mat matHomo;
	refineMatchesWithHomography(keypoints1, keypoints2, 3, matches, matHomo);
	cout << "[Info] Homography T : " << matHomo << endl;
	cout << "[Info] # of matches : " << matches.size() << endl;

	Mat imResult;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, imResult,
		CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));

	// 计算光流  
	vector<uchar> vstatus;
	vector<float> verrs;
	vector<Point2f> points1;
	vector<Point2f> points2;
	KeyPointsToPoints(keypoints1, points1);

	calcOpticalFlowPyrLK(img1, img2, points1, points2, vstatus, verrs);

	Mat imOFKL = img1.clone();
	for (int i = 0; i < vstatus.size(); i++) {
		if (vstatus[i] && verrs[i] < 15) {
			line(imOFKL, points1[i], points2[i], CV_RGB(255, 255, 255), 1, 8, 0);
			circle(imOFKL, points2[i], 3, CV_RGB(255, 255, 255), 1, 8, 0);
		}
	}


	imwrite("opt.jpg", imOFKL);
	imwrite("re1.jpg", imResultOri);
	imwrite("re2.jpg", imResult);

	imshow("Optical Flow", imOFKL);
	imshow("origin matches", imResultOri);
	imshow("refined matches", imResult);
	waitKey();

	return -1;
}

bool refineMatchesWithHomography(
	const std::vector<cv::KeyPoint>& queryKeypoints,
	const std::vector<cv::KeyPoint>& trainKeypoints,
	float reprojectionThreshold, std::vector<cv::DMatch>& matches,
	cv::Mat& homography) {
	const int minNumberMatchesAllowed = 8;

	if (matches.size() < minNumberMatchesAllowed)
		return false;

	// Prepare data for cv::findHomography  
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	// Find homography matrix and get inliers mask  
	std::vector<unsigned char> inliersMask(srcPoints.size());
	homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC,
		reprojectionThreshold, inliersMask);

	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) {
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return matches.size() > minNumberMatchesAllowed;
}

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts) {
	for (int i = 0; i < kpts.size(); i++) {
		pts.push_back(kpts[i].pt);
	}

	return;
}
#endif
#include <fstream>  
#include <sstream>  
#include "opencv2/opencv.hpp"  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
using namespace cv;
using namespace std;
bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,const std::vector<cv::KeyPoint>& trainKeypoints,float reprojectionThreshold, std::vector<cv::DMatch>& matches,cv::Mat& homography) 
{
	const int minNumberMatchesAllowed = 8;

	if (matches.size() < minNumberMatchesAllowed)
		return false;

	// Prepare data for cv::findHomography  
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());

	for (size_t i = 0; i < matches.size(); i++) 
	{
		srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	// Find homography matrix and get inliers mask  
	std::vector<unsigned char> inliersMask(srcPoints.size());
	homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC,reprojectionThreshold, inliersMask);

	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) 
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return matches.size() > minNumberMatchesAllowed;
}

void KeyPointsToPoints(vector<KeyPoint> kpts, vector<Point2f> &pts) 
{
	for (int i = 0; i < kpts.size(); i++) 
	{
		pts.push_back(kpts[i].pt);
	}
	return;
}
vector<Mat> descriptor;
vector<vector<KeyPoint>> keypoints; //////////////// 提取SIFT特征
SiftFeatureDetector temp_sift(200);     //////////////// 构造SIFT特征检测器
SiftDescriptorExtractor temp_siftDesc;  //////////////// 构造SIFT描述子提取器
void tempfunc()
{
	char filename[10];
	for (int i = 1; i <= 100; i++)
	{
		Mat temp_descriptor;
		vector<KeyPoint > temp_keypoint;
		sprintf_s(filename, "%d.jpg", i);
		Mat temp_img = imread(filename, 0);
		Mat size_img;
		resize(temp_img, size_img, Size(128, 90));
		temp_sift.detect(size_img, temp_keypoint);
		keypoints.push_back(temp_keypoint);
		temp_siftDesc.compute(size_img, temp_keypoint, temp_descriptor);
		descriptor.push_back(temp_descriptor);
	}
}
#if 1
int main(int argc, char *argv[])
{
	// 以下两图比之
	// 输入两张要匹配的图
	Directory dir;

	Mat tempimg1 = imread("9.jpg", 0);
	string path1 = "./8s/";
	string exten1 = "*.jpg";
	bool addPath1 = false;
	vector<string> filenames = dir.GetListFiles(path1, exten1, addPath1);
	vector<int> nummatcher;
	vector<int> numgoodmatcher;
	vector<int> distancematcher;
	vector<string> imgstr;
	int count = 0;
	for (int i = 0; i < filenames.size(); i++)
	{
		string imgPath;
	/*	filenames[i] = "6139.jpg";*/
		imgPath = path1 + filenames[i];
		Mat tempimg2 = imread(imgPath, 0);
		Mat image1, image2;
		resize(tempimg1, image1, Size(80, 70));
		resize(tempimg2, image2, Size(80, 70));
// 		 		namedWindow("Right Image");
// 		 		imshow("Right Image", image1);
// 		 		namedWindow("Left Image");
// 		 		imshow("Left Image", image2);

		// 存放特征点的向量
		vector<KeyPoint> keypoint1;
		vector<KeyPoint> keypoint2;

		//////////////////////////////////////////////////////// 构造SURF特征检测器
		SiftFeatureDetector sift(200); // 阈值

		// 对两幅图分别检测SURF特征
		sift.detect(image1, keypoint1);
		sift.detect(image2, keypoint2);

		// 输出带有详细特征点信息的两幅图像
// 		Mat imageSURF;
// 		drawKeypoints(image1, keypoint1, imageSURF, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 		namedWindow("Right SURF Features");
// 		imshow("Right SURF Features", imageSURF);
// 		drawKeypoints(image2, keypoint2, imageSURF, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 		namedWindow("Left SURF Features");
// 		imshow("Left SURF Features", imageSURF);

		//////////////////////////////////////////////////////// 构造SURF描述子提取器
		SiftDescriptorExtractor siftDesc;

		// 对两幅图像提取SURF描述子
		Mat descriptor1, descriptor2;
		siftDesc.compute(image1, keypoint1, descriptor1);
		siftDesc.compute(image2, keypoint2, descriptor2);

		///////////////////////////////////////////////////////// 构造匹配器
		BruteForceMatcher< cv::L2<float> > matcher;
		/*FlannBasedMatcher matcher;*/

		// 将两张图片的描述子进行匹配
		vector<DMatch> matches;
		vector<vector<DMatch>> m_knnMatches;
		vector<DMatch> good_matches;

		matcher.match(descriptor1, descriptor2, matches);

		double max_dist = 0;
		double min_dist = 1000;
		//快速计算关键点之间的最大和最小距离
		for (int i = 0; i < descriptor1.rows; i++)
		{
			double dist = matches[i].distance;
			distancematcher.push_back(dist);
			if (dist < min_dist)
			{
				min_dist = dist;
			}
			if (dist > max_dist)
			{
				max_dist = dist;
			}
		}

		for (int i = 0; i < descriptor1.rows; i++)
		{
			if (matches[i].distance < 2 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		////////////////////////////////////////////////////////////
		Mat matHomo;
		refineMatchesWithHomography(keypoint1, keypoint2, 3, matches, matHomo);

		nummatcher.push_back(matches.size());
		cout << "[Info] # of matches : " << matches.size() << endl;

		Mat imResult;
 		drawMatches(image1, keypoint1, image2, keypoint2, good_matches, imResult, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
 		imshow("imResult", imResult);
		waitKey();
		cout << "good_matches size =  " << good_matches.size() << endl;
		
		numgoodmatcher.push_back(good_matches.size());
		//////////////////////////////////////////////////////////////
		
		imgstr.push_back(filenames[i]);
		count++;

	}
	FileStorage fs("numgoodmatcher.txt", FileStorage::WRITE);
	fs << "numgoodmatcher" << numgoodmatcher;
	fs << "numgoodmatcher" << distancematcher;
	fs << "numgoodmatcher" << imgstr;
	fs.release();
	FileStorage fss("nummatcher.txt", FileStorage::WRITE);
	fss << "nummatcher" << nummatcher;
	fss << "imgname" << imgstr;
	fss.release();
	cout << "img counts =  " << count << endl;
	return 1;
}
#endif
#if 0
int main(int argc, char *argv[])
{
	Mat tempimg1 = imread("004.jpg", 0);

	Mat image1;
	resize(tempimg1, image1, Size(128, 90));
 
// 	namedWindow("Right Image");
// 	imshow("Right Image", image1);

	// 存放特征点的向量
	vector<KeyPoint> keypoint1;
	temp_sift.detect(image1, keypoint1);


	// 输出带有详细特征点信息的两幅图像
	Mat imageSift;
	drawKeypoints(image1, keypoint1, imageSift, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 	namedWindow("sift Features");
// 	imshow("sift Features", imageSift);

	// 提取sift特征
	Mat descriptor1;
	temp_siftDesc.compute(image1, keypoint1, descriptor1);

	///////////////////////////////////////////////////////// 构造匹配器
	BruteForceMatcher< cv::L2<float> > matcher;
	vector<DMatch> matches;
	float Maxsize = 0;
	int picture = 0;
	tempfunc();
	for (int i = 0; i < 10;i++)
	{
		vector<DMatch> temp_matche;
		matcher.match(descriptor1, descriptor[i], temp_matche);
		matcher.match(descriptor1, descriptor[i], temp_matche);
 		Mat matHomo;
  		refineMatchesWithHomography(keypoint1, keypoints[i], 3, temp_matche, matHomo);
 		cout << "[Info] # of matches : " << temp_matche.size() << endl;
		if (temp_matche.size()> Maxsize)
		{
			matches.clear();
			Maxsize = temp_matche.size();
			picture = i+1;
			matches = temp_matche;
		}
	}

	////////////////////////////////////////////////////////////
	char picturename[10];
	sprintf_s(picturename, "%d.jpg", picture);
	Mat image2 = imread(picturename, 0);
// 	Mat imResultOri;
// 	drawMatches(image1, keypoint1, image2, temp_keypoint[picture], matches, imResultOri, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
// 	cout << "[Info] # of matches : " << matches.size() << endl;
// 	imshow("imResultOri", imResultOri);
// 	Mat matHomo;
// 	refineMatchesWithHomography(keypoint1, temp_keypoint[picture], 3, matches, matHomo);
// 	cout << "[Info] Homography T : " << matHomo << endl;
// 	cout << "[Info] # of matches : " << matches.size() << endl;

	Mat imResult;
	drawMatches(image1, keypoint1, image2, keypoints[picture], matches, imResult, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
	imshow("imResult", imResult);

	waitKey();
	return 1;
}
#endif

#if 0
void BOWKeams(const Mat& img, const vector<KeyPoint>& Keypoints,const Mat& Descriptors, Mat& centers)
{
	//BOW的kmeans算法聚类;
	BOWKMeansTrainer bowK(10,
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1), 3, 2);
	centers = bowK.cluster(Descriptors);
	cout << endl << "< cluster num: " << centers.rows << " >" << endl;

	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> matches;
	descriptorMatcher->match(Descriptors, centers, matches);//const Mat& queryDescriptors, const Mat& trainDescriptors第一个参数是待分类节点，第二个参数是聚类中心;
	Mat demoCluster;
	img.copyTo(demoCluster);

	//为每一类keyPoint定义一种颜色
	Scalar color[] = { CV_RGB(255, 255, 255),
		CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0, 255),
		CV_RGB(255, 255, 0), CV_RGB(255, 0, 255), CV_RGB(0, 255, 255),
		CV_RGB(123, 123, 0), CV_RGB(0, 123, 123), CV_RGB(123, 0, 123) };


	for (vector<DMatch>::iterator iter = matches.begin(); iter != matches.end(); iter++)
	{
		cout << "< descriptorsIdx:" << iter->queryIdx << "  centersIdx:" << iter->trainIdx
			<< " distincs:" << iter->distance << " >" << endl;
		Point center = Keypoints[iter->queryIdx].pt;
		circle(demoCluster, center, 2, color[iter->trainIdx], -1);
	}
	putText(demoCluster, "KeyPoints Clustering: 一种颜色代表一种类型",
		cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar::all(-1));
	imshow("KeyPoints Clusrtering", demoCluster);
	waitKey();
}
void dense_SIFT_BoW(Mat img_raw, Mat &featuresUnclustered)
{

	Mat descriptors; // Store our dense SIFT descriptors.
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
	//DenseFeatureDetector detector(12.f, 1, 0.1f, 10);
	//DenseFeatureDetector detector;
	detector->detect(img_raw, keypoints);

	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

	descriptorExtractor->compute(img_raw, keypoints, descriptors);
	Mat center;
    BOWKeams(img_raw, keypoints, descriptors, center);
    //descriptors.setTo(0, descriptors < 0);
    //descriptors = descriptors.reshape(0, 1);

	featuresUnclustered.push_back(center);
}
int main(int argc, char *argv[])
{
	Mat tempimg1 = imread("1.jpg", 0);
	Mat tempimg2 = imread("13.jpg", 0);

	Mat image1, image2;
	resize(tempimg1, image1, Size(128, 90));
	resize(tempimg2, image2, Size(128, 90));

	initModule_nonfree();

	//sift关键点检测  
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	detector.detect(image1, keypoint1);
	detector.detect(image2, keypoint2);
	//sift关键点描述，角度，强度等  
	SiftDescriptorExtractor extractor;
	Mat descriptor_f, descriptor_s;
	extractor.compute(image1, keypoint1, descriptor_f);
	extractor.compute(image2, keypoint2, descriptor_s);


	int clusterNum = 26;
	//clusterNum代表有多少词  
	BOWKMeansTrainer trainer(clusterNum);
	trainer.add(descriptor_f);
	trainer.add(descriptor_s);
	Mat dictionary = trainer.cluster();
	Ptr<DescriptorExtractor> desExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> matchers = DescriptorMatcher::create("BruteForce");

	BOWImgDescriptorExtractor bowDE(desExtractor, matchers);
	bowDE.setVocabulary(dictionary);

	Mat BOWdescriptor_f, BOWdescriptor_s;
	//sift关键点检测  
	vector<KeyPoint> keyPoints_f, keyPoints_s;
	/*SiftFeatureDetector detector;*/
	detector.detect(image1, keyPoints_f);
	detector.detect(image2, keyPoints_s);
	//BOWdecriptor表示每个图像的bow码本，即直方图，大小为1*clusterNum  
	Ptr<BOWImgDescriptorExtractor> bowExtractor;
	bowDE.compute(dictionary, keyPoints_f, BOWdescriptor_f);
	bowDE.compute(image2, keyPoints_s, BOWdescriptor_s);
	//归一化  
	normalize(BOWdescriptor_f, BOWdescriptor_f, 1.0, 0.0, NORM_MINMAX);
	normalize(BOWdescriptor_s, BOWdescriptor_s, 1.0, 0.0, NORM_MINMAX);


// 	dense_SIFT_BoW(image1, featuresUnclustered_f);
// 	cout << featuresUnclustered_f.size() << endl;
// 	dense_SIFT_BoW(image2, featuresUnclustered_s);
// 	cout << featuresUnclustered_s.size() << endl;

// 	// the number of bags            //Construct BOW k-means trainer
// 	int dictionarySize = 200;
// 
// 	//define term criteria
// 	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
// 
// 	// retry number
// 	int retries = 1;
// 
// 	//necessary flags
// 	int flags = KMEANS_PP_CENTERS;
// 
// 	//Create the BOW trainer
// 	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
// 
// 	//cluster the feature vectors
// 	Mat dictionary_f = bowTrainer.cluster(featuresUnclustered_f);
// 	Mat dictionary_s = bowTrainer.cluster(featuresUnclustered_s);
	///////////////////////////////////////////////////////////////////////
	BruteForceMatcher<L2<float> > matcher;
	vector<DMatch> matches;
	matcher.match(BOWdescriptor_f, BOWdescriptor_s, matches);
	cout << "[Info] # of matches : " << matches.size() << endl;


#if 0
	// 存放特征点的向量
	vector<KeyPoint> keypoint1;
	vector<KeyPoint> keypoint2;
	//////////////////////////////////////////////////////// 构造SURF特征检测器
	SiftFeatureDetector sift(200); // 阈值

	// 对两幅图分别检测SURF特征
	sift.detect(image1, keypoint1);
	sift.detect(image2, keypoint2);

	// 输出带有详细特征点信息的两幅图像
// 	Mat imageSURF;
// 	drawKeypoints(image1, keypoint1, imageSURF, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 	namedWindow("Right SURF Features");
// 	imshow("Right SURF Features", imageSURF);
// 	drawKeypoints(image2, keypoint2, imageSURF, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
// 	namedWindow("Left SURF Features");
// 	imshow("Left SURF Features", imageSURF);

	//////////////////////////////////////////////////////// 构造SURF描述子提取器
	SiftDescriptorExtractor siftDesc;

	// 对两幅图像提取SURF描述子
	Mat descriptor1, descriptor2;
	siftDesc.compute(image1, keypoint1, descriptor1);
	siftDesc.compute(image2, keypoint2, descriptor2);

	///////////////////////////////////////////////////////// 构造匹配器
	BruteForceMatcher< cv::L2<float> > matcher_s;

	// 将两张图片的描述子进行匹配，只选择25个最佳匹配
	vector<cv::DMatch> matche_s;
	matcher_s.match(descriptor1, descriptor2, matche_s);
	////////////////////////////////////////////////////////////

	Mat imResultOri;
	drawMatches(image1, keypoint1, image2, keypoint2, matche_s, imResultOri, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
	cout << "[Info] # of matches : " << matche_s.size() << endl;
	imshow("imResultOri", imResultOri);
	Mat matHomo;
	refineMatchesWithHomography(keypoint1, keypoint2, 3, matche_s, matHomo);
	cout << "[Info] Homography T : " << matHomo << endl;
	cout << "[Info] # of matches : " << matche_s.size() << endl;

	Mat imResult;
	drawMatches(image1, keypoint1, image2, keypoint2, matche_s, imResult, CV_RGB(0, 255, 0), CV_RGB(0, 255, 0));
	imshow("imResult", imResult);
	//////////////////////////////////////////////////////////////
#endif
	waitKey();
	return 1;
}
#endif