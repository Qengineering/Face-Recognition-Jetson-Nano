#include "TBlur.h"
#include <iostream>
//----------------------------------------------------------------------------------------
//
// Created by Q-engineering 2020/9/25
//
//----------------------------------------------------------------------------------------
using namespace cv;
using namespace std;
//----------------------------------------------------------------------------------------
TBlur::TBlur()
{
    cx = 64;
    cy = 64;
    Block = 52;     //low frequency blocker
}
//----------------------------------------------------------------------------------------
TBlur::~TBlur()
{
    //dtor
}
//----------------------------------------------------------------------------------------
double TBlur::Execute(cv::Mat &frame)
{
    //resize to a small picture
    cv::Mat SmallFrame;
    cv::resize(frame, SmallFrame, Size(128,128));

    // go to black and white
    cv::Mat BWFrame;
    cvtColor(SmallFrame,BWFrame,cv::COLOR_BGR2GRAY);

    // go to float
    Mat fImage;
    BWFrame.convertTo(fImage, CV_32FC1);

    // FFT
    Mat fourierTransform;
    cv::dft(fImage, fourierTransform, DFT_SCALE|DFT_COMPLEX_OUTPUT);

    //center low frequencies in the middle
    //by shuffling the quadrants.
    Mat q0(fourierTransform, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
    Mat q1(fourierTransform, Rect(cx, 0, cx, cy));      // Top-Right
    Mat q2(fourierTransform, Rect(0, cy, cx, cy));      // Bottom-Left
    Mat q3(fourierTransform, Rect(cx, cy, cx, cy));     // Bottom-Right

    Mat tmp;                                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // Block the low frequencies
    // #define BLOCK could also be a argument on the command line of course
    fourierTransform(Rect(cx-Block,cy-Block,2*Block,2*Block)).setTo(0);

    //shuffle the quadrants to their original position
    Mat orgFFT;
    fourierTransform.copyTo(orgFFT);
    Mat p0(orgFFT, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
    Mat p1(orgFFT, Rect(cx, 0, cx, cy));      // Top-Right
    Mat p2(orgFFT, Rect(0, cy, cx, cy));      // Bottom-Left
    Mat p3(orgFFT, Rect(cx, cy, cx, cy));     // Bottom-Right

    p0.copyTo(tmp);
    p3.copyTo(p0);
    tmp.copyTo(p3);

    p1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
    p2.copyTo(p1);
    tmp.copyTo(p2);

    // IFFT
    Mat invFFT;
    dft(orgFFT, invFFT, DFT_INVERSE|DFT_REAL_OUTPUT);

    //img_fft = 20*numpy.log(numpy.abs(img_fft))
    Mat logFFT;
    cv::abs(invFFT);
    cv::log(invFFT,logFFT);
    logFFT *= 20;

    //result = numpy.mean(img_fft)
    cv::Scalar result= cv::mean(logFFT);
    return result.val[0];
}
//----------------------------------------------------------------------------------------

