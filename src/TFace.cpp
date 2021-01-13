#include "TFace.h"
//----------------------------------------------------------------------------------------
TFace::TFace()
{
    //ctor
}
//----------------------------------------------------------------------------------------
TFace::~TFace()
{
    //dtor
}
//----------------------------------------------------------------------------------------
void TFace::Process(std::vector<FaceObject> Faces)
{
    size_t i;

    for(i = 0; i< Faces.size(); i++) {
        float x_  =  Faces[i].rect.x1;
        float y_  =  Faces[i].rect.y1;
        float x2_ =  faceInfo[i].x2;
        float y2_ =  faceInfo[i].y2;
        int x = (int) x_ ;
        int y = (int) y_;
        int x2 = (int) x2_;
        int y2 = (int) y2_;
        struct TLiveFaceBox  live_box={x_,y_,x2_,y2_} ;

        cv::rectangle(result_cnn, Point(x*ratio_x, y*ratio_y), Point(x2*ratio_x,y2*ratio_y), cv::Scalar(0, 0, 255), 2);
        // Perspective Transformation
        float v2[5][2] =
                {{faceInfo[i].ppoint[0], faceInfo[i].ppoint[5]},
                {faceInfo[i].ppoint[1], faceInfo[i].ppoint[6]},
                {faceInfo[i].ppoint[2], faceInfo[i].ppoint[7]},
                {faceInfo[i].ppoint[3], faceInfo[i].ppoint[8]},
                {faceInfo[i].ppoint[4], faceInfo[i].ppoint[9]},
                };


        // compute the turning angle
        angle = count_angle(v2);

                static std::string hi_name;
                static std::string liveface;
                static int stranger,close_enough;


/****************************jump*****************************************************/
                if (count%jump==0){
                cv::Mat dst(5, 2, CV_32FC1, v2);
                memcpy(dst.data, v2, 2 * 5 * sizeof(float));

                cv::Mat m = FacePreprocess::similarTransform(dst, src);
                cv::Mat aligned = frame.clone();
                cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
                resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);


                //set to 1 if you want to record your image
                if (record_face) {
                    imshow("aligned face", aligned);
                    waitKey(2000);
                    imwrite(project_path+ format("/img/%d.jpg", count), aligned);
                }
                //features of camera image
                cv::Mat fc2 = reco.getFeature(aligned);

                // normalize
                fc2 = Zscore(fc2);

                //the similarity score
                vector<double> score_;
                for (unsigned int compare_ = 0; compare_ < image_number; ++ compare_){
                    score_.push_back(CosineDistance(fc1[compare_], fc2));
                }
                int maxPosition = max_element(score_.begin(),score_.end()) - score_.begin();
                current=score_[maxPosition];
                score_.clear();
                sprintf(string, "%.4f", current);

                 if (current >= face_thre && y2-y>= distance_threshold){
                            //put name
                            int slant_position=image_names[maxPosition].rfind ('/');
                            cv::String name = image_names[maxPosition].erase(0,slant_position+1);
                            name=name.erase( name.length()-4, name.length()-1);
                            hi_name="Hi,"+name;
                            putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                            cout<<name<<endl;
                            //determin whethe it is a fake face
                            confidence=live.Detect(frame,live_box);

                            sprintf(string1, "%.4f", confidence);
                            cv::putText(result_cnn,string1, Point(x*ratio_x, y2*ratio_y+20), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0,255,255),2);
                            if (confidence<=true_thre)
                                    {putText(result_cnn, "Fake face!!", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
                                    liveface="Fake face!!";
                                    }
                                else
                                    {putText(result_cnn, "True face", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);
                                    liveface="True face";
                                    }
                                cout<<liveface<<endl;
                                stranger=0;
                                close_enough=1;
                                }

                else if(current >= face_thre && y2-y < distance_threshold){
                                //put name
                                int slant_position=image_names[maxPosition].rfind ('/');
                                cv::String name = image_names[maxPosition].erase(0,slant_position+1);
                                name=name.erase( name.length()-4, name.length()-1);
                                hi_name="Hi,"+name;
                                putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                                //Ask be closer to avoid mis-reco
                                putText(result_cnn, "Closer please", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                                cout<<"Closer please"<<endl;
                                stranger=0;
                                close_enough=0;
                    }
                else {
                            putText(result_cnn, "Stranger", cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
                            cout<<"Stranger"<<endl;
                            stranger=1;
                }
                 //highlight the significant landmarks on face
                for (int j = 0; j < 5; j += 1) {
                    if (j == 0 or j == 3) {
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(0, 255, 0),
                                FILLED, LINE_AA);
                    } else if (j==2){
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(255, 0, 0),
                                FILLED, LINE_AA);
                    }
                        else {
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(0, 0, 255),
                                FILLED, LINE_AA);
                        }
                }
                cv::putText(result_cnn,string, Point(x*ratio_x, y2*ratio_y), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);


                }
                else{
                    if(stranger)
                    {
                         putText(result_cnn, "Stranger", cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
                    }
                    else if(close_enough)
                    {
                         putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                         putText(result_cnn,string1,Point(x*ratio_x, y2*ratio_y+20), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0,255,255),2);
                         if (liveface.length()==9)
                            putText(result_cnn, liveface, cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);
                         else
                            putText(result_cnn, liveface, cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 0, 255),2);
                    }
                    else
                    {
                       putText(result_cnn, hi_name, cv::Point(5, 60), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                       putText(result_cnn, "Closer please", cv::Point(5, 80), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(0, 255, 255),2);
                    }

                if (count==10*jump-1) count=0;
                 //highlight the significant landmarks on face
                for (int j = 0; j < 5; j += 1) {
                    if (j == 0 or j == 3) {
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(0, 255, 0),
                                FILLED, LINE_AA);
                    } else if (j==2){
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(255, 0, 0),
                                FILLED, LINE_AA);
                    }
                        else {
                        cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                Scalar(0, 0, 255),
                                FILLED, LINE_AA);
                        }
            }
                cv::putText(result_cnn,string, Point(x*ratio_x, y2*ratio_y), cv::FONT_HERSHEY_SIMPLEX,0.75, cv::Scalar(255, 255, 0),2);


                }

            }

//            t = ((double) cv::getTickCount() - t) / (cv::getTickFrequency());
//            fps.push_back(1.0/t);
//            int fpsnum_= fps.size();
//            float fps_mean;
//            //compute average fps value
//            if(fpsnum_<=30){
//                sum_fps = std::accumulate(std::begin(fps), std::end(fps), 0.0);
//                fps_mean = sum_fps /  fpsnum_;
//
//            }
//            else{
//                sum_fps = std::accumulate(std::end(fps)-30, std::end(fps), 0.0);
//                fps_mean = sum_fps /  30;
//                if(fpsnum_>=300) fps.clear();
//
//            }
//            result_cnn = draw_conclucion("FPS: ", fps_mean, result_cnn, 20);//20
//            result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 40);//65
//
//
//            cv::imshow("image", result_cnn);
//            cv::waitKey(1);

        }
}
//----------------------------------------------------------------------------------------
