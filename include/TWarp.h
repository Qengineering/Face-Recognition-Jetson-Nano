#ifndef TFACE_H
#define TFACE_H
#include "TRetina.h"
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
class TWarp
{
private:
    int     MatrixRank(cv::Mat M);
    cv::Mat VarAxis0(const cv::Mat &src);
    cv::Mat MeanAxis0(const cv::Mat &src);
    cv::Mat ElementwiseMinus(const cv::Mat &A,const cv::Mat &B);
    cv::Mat SimilarTransform(cv::Mat src,cv::Mat dst);
protected:
public:
    double Angle;

    TWarp();
    virtual ~TWarp();

    cv::Mat Process(cv::Mat &SmallFrame,FaceObject& Obj);
};
//----------------------------------------------------------------------------------------
#endif // TFACE_H
