#include "TArcface.h"
//----------------------------------------------------------------------------------------
//
// Created by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
TArcFace::TArcFace(void)
{
    net.load_param("./models/mobilefacenet/mobilefacenet.param");
    net.load_model("./models/mobilefacenet/mobilefacenet.bin");
}
//----------------------------------------------------------------------------------------
TArcFace::~TArcFace()
{
    this->net.clear();
}
//----------------------------------------------------------------------------------------
//    This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
//    original distribution in order to make two feature more distinguishable.
//    mean value is set to 0 and std is set to 1
cv::Mat TArcFace::Zscore(const cv::Mat &fc)
{
    cv::Mat mean, std;
    meanStdDev(fc, mean, std);
    return((fc - mean) / std);
}
//----------------------------------------------------------------------------------------
cv::Mat TArcFace::GetFeature(cv::Mat img)
{
    vector<float> feature;
    //cv to NCNN
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(this->feature_dim);
    for (int i = 0; i < this->feature_dim; i++) feature[i] = out[i];
    //normalize(feature);
    cv::Mat feature__=cv::Mat(feature,true);
    return Zscore(feature__);
}
//----------------------------------------------------------------------------------------
