#include "image_matcher.h"
#include <opencv2/xfeatures2d.hpp>

ImageMatcher::ImageMatcher() {}

ImageMatcher::~ImageMatcher() {}

ImageMatcher::ImageMatcher(const cv::Mat& img0, const cv::Mat& img1,
	FeatureType method1, MatcherType method2, int knn)
	:query_image_(img0), refer_image_(img1), feature_method_(method1),
	matcher_method_(method2) {

	ExtractFeatures();
	MatchFeatures(knn);
}

void ImageMatcher::ExtractFeatures() {

	cv::Ptr<cv::Feature2D> feature;

	int i = 0;
	int jhist = 0, jori = 0;
	cv::Mat temp;
	switch (feature_method_)
	{
	case FEATURE_SIFT:
		feature = cv::xfeatures2d::SIFT::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);
		break;
	case FEATURE_SURF:
		feature = cv::xfeatures2d::SURF::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);
		break;
	case FEATURE_ORB:
		feature = cv::ORB::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);
		break;
	case FEATURE_AKAZE:
		feature = cv::AKAZE::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);
		break;
	case FEATURE_ROOTSIFT:
		feature = cv::xfeatures2d::SIFT::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);

		for (i = 0; i < query_des_.rows; ++i) {
			cv::normalize(query_des_.row(i), temp, 1, cv::NORM_L1);
			cv::sqrt(temp, temp);
			temp.row(0).copyTo(query_des_.row(i));
		}

		for (i = 0; i < refer_des_.rows; ++i) {
			cv::normalize(refer_des_.row(i), temp, 1, cv::NORM_L1);
			cv::sqrt(temp, temp);
			temp.row(0).copyTo(refer_des_.row(i));
		}

		break;
	case FEATURE_HALFSIFT:
		feature = cv::xfeatures2d::SIFT::create();
		feature->detectAndCompute(query_image_, cv::Mat(), query_kpts_, query_des_);
		feature->detectAndCompute(refer_image_, cv::Mat(), refer_kpts_, refer_des_);

		for (i = 0; i < query_des_.rows; ++i) {
			for (jhist = 0; jhist < 16; ++jhist) {
				for (jori = 0; jori < 4; ++jori) {
					temp = (query_des_.row(i).col(jhist * 8 + jori) +
						query_des_.row(i).col(jhist * 8 + jori + 4));
					temp.copyTo(query_des_.row(i).col(jhist * 8 + jori));
					temp.copyTo(query_des_.row(i).col(jhist * 8 + jori + 4));
				}
			}
		}

		for (i = 0; i < refer_des_.rows; ++i) {
			for (jhist = 0; jhist < 16; ++jhist) {
				for (jori = 0; jori < 4; ++jori) {
					temp = (refer_des_.row(i).col(jhist * 8 + jori) +
						refer_des_.row(i).col(jhist * 8 + jori + 4));
					temp.copyTo(refer_des_.row(i).col(jhist * 8 + jori));
					temp.copyTo(refer_des_.row(i).col(jhist * 8 + jori + 4));
				}
			}
		}
		break;
	default:
		break;
	}

	// Make sure the types of the descriptors support the FlannBasedMatcher.
	query_des_.convertTo(query_des_, CV_32F);
	refer_des_.convertTo(refer_des_, CV_32F);
}

void ImageMatcher::MatchFeatures(int knn) {

	cv::Ptr<cv::DescriptorMatcher> matcher;
	switch (matcher_method_)
	{
	case MATCHER_BF:
		matcher = cv::DescriptorMatcher::create("BruteForce");
		break;
	case MATCHER_FLANN:
		matcher = cv::DescriptorMatcher::create("FlannBased");
		break;
	}
	matcher->knnMatch(query_des_, refer_des_, matches_, knn);

}

void ImageMatcher::GetKeyPoints(std::vector<cv::KeyPoint>& key_points0,
	std::vector<cv::KeyPoint>& key_points1) const {
	key_points0 = query_kpts_;
	key_points1 = refer_kpts_;
}

void ImageMatcher::GetMatches(std::vector<std::vector<cv::DMatch> >& matches) const {
	matches = matches_;
}