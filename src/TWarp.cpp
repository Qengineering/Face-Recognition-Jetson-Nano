#include "TWarp.h"
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
// Calculating the turning angle of face
//----------------------------------------------------------------------------------------
inline double count_angle(float landmark[5][2])
{
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180.0 / M_PI;
    return angle;
}
//----------------------------------------------------------------------------------------
// TWarp
//----------------------------------------------------------------------------------------
TWarp::TWarp()
{
    //ctor
}
//----------------------------------------------------------------------------------------
TWarp::~TWarp()
{
    //dtor
}
//----------------------------------------------------------------------------------------
cv::Mat TWarp::MeanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2
    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i++){
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++){
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}
//----------------------------------------------------------------------------------------
cv::Mat TWarp::ElementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}
//----------------------------------------------------------------------------------------
int TWarp::MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}
//----------------------------------------------------------------------------------------
cv::Mat TWarp::VarAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = ElementwiseMinus(src,MeanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return MeanAxis0(temp_);

}
//----------------------------------------------------------------------------------------
//    References
//    ----------
//    .. [1] "Least-squares estimation of transformation parameters between two
//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
//
//    Anthor:Jack Yu
cv::Mat TWarp::SimilarTransform(cv::Mat src,cv::Mat dst)
{
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;

    // the SVD function in opencv differ from scipy .
    cv::SVD::compute(A, S,U, V);

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = VarAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    float scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t(); //src_mean.T
    cv::Mat temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}
//----------------------------------------------------------------------------------------
cv::Mat TWarp::Process(cv::Mat& SmallFrame,FaceObject& Obj)
{
    // gt face landmark
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}
    };
    static cv::Mat src(5, 2, CV_32FC1, v1);
    memcpy(src.data, v1, 2*5*sizeof(float));

    // Perspective Transformation
    float v2[5][2] ={
        {Obj.landmark[0].x, Obj.landmark[0].y},
        {Obj.landmark[1].x, Obj.landmark[1].y},
        {Obj.landmark[2].x, Obj.landmark[2].y},
        {Obj.landmark[3].x, Obj.landmark[3].y},
        {Obj.landmark[4].x, Obj.landmark[4].y},
    };
    cv::Mat dst(5, 2, CV_32FC1, v2);
    memcpy(dst.data, v2, 2*5*sizeof(float));

    // compute the turning angle
    Angle = count_angle(v2);

    cv::Mat aligned = SmallFrame.clone();
    cv::Mat m = SimilarTransform(dst, src);
    cv::warpPerspective(SmallFrame, aligned, m, cv::Size(96, 112), cv::INTER_LINEAR);
    resize(aligned, aligned, cv::Size(112, 112), 0, 0, cv::INTER_LINEAR);

    return aligned;
}
//----------------------------------------------------------------------------------------
