/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/
#ifndef MVO_FEATURE_H
#define MVO_FEATURE_H

#include <opencv2/opencv.hpp>

namespace mvo{

    void
    featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                    std::vector<uchar>& status){

//this function automatically gets rid of points for which tracking fails

        std::vector<float> err;
        cv::Size winSize = cv::Size(21, 21);
        cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

        //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for(int i = 0; i < status.size(); i++){
            cv::Point2f pt = points2.at(i - indexCorrection);
            if((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)){
                if((pt.x < 0) || (pt.y < 0)){
                    status.at(i) = 0;
                }
                points1.erase(points1.begin() + (i - indexCorrection));
                points2.erase(points2.begin() + (i - indexCorrection));
                indexCorrection++;
            }
        }
    }

    //uses FAST as of now, modify parameters as necessary
    void featureDetection(cv::Mat img_1, std::vector<cv::Point2f>& points1){
        std::vector<cv::KeyPoint> keypoints_1;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        cv::FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
        cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());
    }
} // namespace mvo
#endif //MVO_FEATURE_H
