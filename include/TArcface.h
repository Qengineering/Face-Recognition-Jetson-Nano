#ifndef TARCFACE_H
#define TARCFACE_H

#include <cmath>
#include <vector>
#include <string>
#include "net.h"
#include <opencv2/highgui.hpp>
//----------------------------------------------------------------------------------------
//
// Created by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
using namespace std;

class TArcFace {
private:
    ncnn::Net net;
    const int feature_dim = 128;
    cv::Mat Zscore(const cv::Mat &fc);
public:
    TArcFace(void);
    ~TArcFace(void);

    cv::Mat GetFeature(cv::Mat img);
};

#endif
