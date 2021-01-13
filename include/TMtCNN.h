#ifndef TMTCNN_H
#define TMTCNN_H
//
// Created by Lonqi on 2017/11/18.
//
// Modified by Q-engineering 2020/12/28
//
#include "net.h"
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <opencv2/highgui.hpp>
#include "TRetina.h"

using namespace std;
//----------------------------------------------------------------------------------------
struct face_landmark
{
	float x[5];
	float y[5];
};
//----------------------------------------------------------------------------------------
struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    face_landmark landmark;
    float regreCoord[4];
};
//----------------------------------------------------------------------------------------
class TMtCNN {
private:
	const float threshold[3] = { 0.8f, 0.8f, 0.6f };
	int minsize = 40;
	const float pre_facetor = 0.709f;
private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, float scale);
	void nmsTwoBoxs(vector<Bbox> &boundingBox_, vector<Bbox> &previousBox_, const float overlap_threshold, string modelname = "Union");
    void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname="Union");
    void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);

	void PNet(float scale);
    void PNet(void);
    void RNet(void);
    void ONet(void);

    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;

    const float nms_threshold[3] = {0.5f, 0.7f, 0.7f};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
	const int MIN_DET_SIZE = 12;
	std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    int img_w, img_h;
public:
	TMtCNN(void);
    ~TMtCNN(void);

	void SetMinFace(int minSize);
    void detect(const cv::Mat& bgr,std::vector<FaceObject> &Faces);
};
//----------------------------------------------------------------------------------------
#endif // TMTCNN_H
