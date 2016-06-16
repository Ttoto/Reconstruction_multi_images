#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QDebug"


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <qprocess.h>


#include <iostream>
#include <set>

#include "QFileSystemModel"
#include "QDir"
#include "qdir.h"
#include <unistd.h>
#include <fcntl.h>

#include "matching.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cv::initModule_nonfree();
    ui->path->setText("/home/sy/set");
    on_path_returnPressed();

    cloud_.reset (new PointCloudT);
    // The number of points in the cloud
    cloud_->resize (1000);

    K = (Mat_<double>(3,3) << 1000, 0, 640,
         0, 1000, 360,
         0, 0, 1);

    distcoeff = (Mat_<double>(5,1) << 0.0, 0.0, 0.0, 0, 0);

    invert(K, Kinv);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_path_returnPressed()
{

    QFileSystemModel *FSM;
    QStringList sImageFilters;
    QDir d;
    QString sPath;

    sImageFilters << "*.png";

    //Show files in the tree view
    sPath = ui->path->text();

    FSM = new QFileSystemModel(this);
    FSM->setFilter(QDir::NoDotAndDotDot | QDir::AllEntries);
    FSM->setRootPath(sPath);
    FSM->setNameFilters(sImageFilters);

    ui->file_browser->setModel(FSM);
    ui->file_browser->setRootIndex(FSM->index(sPath));
    ui->file_browser->hideColumn(1);
    ui->file_browser->hideColumn(2);
    ui->file_browser->hideColumn(3);


    //initiate the filelist to the image
    d.setPath(ui->path->text());
    d.setFilter( QDir::Files | QDir::NoSymLinks );
    d.setNameFilters(sImageFilters);
    filelist = d.entryInfoList();

    //initiate the dir for the yml files
    ymlFileDir = ui->path->text();
    ymlFileDir.append("/data/");
}


void MainWindow::on_PB_Sift_clicked()
{
    QString ymlFile;
    cv::Mat img_orig;
    cv::Mat img_grey;
    vector<KeyPoint> imgpts;
    Mat descriptors;

    for(int i = 0; i < filelist.size(); i++)
    {
        img_orig.release();
        img_grey.release();

        img_orig = imread(filelist.at(i).absoluteFilePath().toStdString(),CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(img_orig,img_grey,CV_BGR2GRAY);
        imgs_orig.push_back(img_orig);

        ymlFile = ymlFileDir;
        ymlFile.append(filelist.at(i).fileName()).append(".yml");

        imgpts.clear();
        descriptors.release();

        if((access(ymlFile.toStdString().c_str(),F_OK))!=-1)//if the yml file exist just skip the step;
        {
            //restore_descriptors_from_file(ymlFile.toStdString(),imgpts,descriptors);
            cout << ymlFile.toStdString() << " has already exist" << endl;
        }
        else
        {
            matching_get_feature_descriptors(img_grey,imgpts,descriptors);
            save_descriptors_to_file(ymlFile.toStdString(),imgpts,descriptors);
            cout << ymlFile.toStdString() << " has " << imgpts.size()
                 << " points (descriptors " << descriptors.rows << ")" << endl;
        }
    }
}





void MainWindow::reconstruct_first_two_view()
{
    QString ymlFile;

    cv::Matx34d P_first;
    cv::Matx34d P_second;

    P_first = cv::Matx34d(1,0,0,0,
                          0,1,0,0,
                          0,0,1,0);

    Pmats[0] = P_first;

    int first_view = 0;
    int second_view =1;

    imggoodpts1.clear();
    imggoodpts2.clear();
    descriptors1.release();
    descriptors2.release();

    ymlFile = ymlFileDir;
    ymlFile.append(filelist.at(first_view).fileName()).append(".yml");
    cout<<ymlFile.toStdString()<<endl;
    restore_descriptors_from_file(ymlFile.toStdString(),imgpts[first_view],descriptors1);

    ymlFile = ymlFileDir;
    ymlFile.append(filelist.at(second_view).fileName()).append(".yml");
    cout<<ymlFile.toStdString()<<endl;
    restore_descriptors_from_file(ymlFile.toStdString(),imgpts[second_view],descriptors2);

    //matching
    matching_fb_matcher(descriptors1,descriptors2,matches_new);
    matching_good_matching_filter(matches_new);

    //estimating and reconstruction
    FindCameraMatrices(K,Kinv,distcoeff,
                       imgpts[first_view],imgpts[second_view],
                       imggoodpts1,imggoodpts2,
                       P_first,P_second,
                       matches_new,
                       outCloud);

    Pmats[1] = P_second;

    outCloud.clear();
    std::vector<cv::KeyPoint> correspImg1Pt;
    TriangulatePoints(imggoodpts1, imggoodpts2, K, Kinv,distcoeff, P_first, P_second, outCloud, correspImg1Pt);

    for (unsigned int i=0; i<outCloud.size(); i++)
    {
        //cout << "surving" << endl;
        outCloud[i].imgpt_for_img.resize(filelist.size());
        for(int j = 0; j<filelist.size();j++)
        {
            outCloud[i].imgpt_for_img[j] = -1;
        }
        outCloud[i].imgpt_for_img[1] = matches_new[i].trainIdx;
    }

    for(unsigned int i=0;i<outCloud.size();i++)
    {
        outCloud_all.push_back(outCloud[i]);
    }

    outCloud_new = outCloud;
}



/* ------------------------------------------------------------------------- */
/** \fn void FindPoseEstimation()
*
* \brief Find the pose of the camera using a image and several corresponding 3D Points
*
*/
/* ------------------------------------------------------------------------- */
bool MainWindow::FindPoseEstimation(
        cv::Mat_<double>& rvec,
        cv::Mat_<double>& t,
        cv::Mat_<double>& R,
        std::vector<cv::Point3f> ppcloud,
        std::vector<cv::Point2f> imgPoints
        )
{
    if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
        return false;
    }

    vector<int> inliers;

    //use CPU
    double minVal,maxVal;
    cv::minMaxIdx(imgPoints,&minVal,&maxVal);
    cv::solvePnPRansac(ppcloud, imgPoints, K, distcoeff, rvec, t,
                       true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);

    vector<cv::Point2f> projected3D;
    cv::projectPoints(ppcloud, rvec, t, K, distcoeff, projected3D);

    if(inliers.size()==0) { //get inliers
        for(unsigned int i=0;i<projected3D.size();i++) {
            if(norm(projected3D[i]-imgPoints[i]) < 5.0)
                inliers.push_back(i);
        }
    }

    //cv::Rodrigues(rvec, R);
    //visualizerShowCamera(R,t,0,255,0,0.1);

    if(inliers.size() < (double)(imgPoints.size())/5.0) {
        cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
        return false;
    }

    if(cv::norm(t) > 200.0) {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    cv::Rodrigues(rvec, R);
    if(!CheckCoherentRotation(R)) {
        cerr << "rotation is incoherent. we should try a different base view..." << endl;
        return false;
    }

    std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
    return true;
}

/* ------------------------------------------------------------------------- */
/** \fn void on_PB_Reconstruction_clicked()
*
* \brief Reconstruction all images
* img1 2 3
* step1:
*
*/
/* ------------------------------------------------------------------------- */
void MainWindow::on_PB_Reconstruction_clicked()
{
    on_PB_Sift_clicked();

    cv::Mat_<double> rvec(1,3);
    vector<KeyPoint> imgpts1_tmp;
    vector<KeyPoint> imgpts2_tmp;

    imgpts.resize(filelist.size());

    reconstruct_first_two_view();

    cv::Mat_<double> t = (cv::Mat_<double>(1,3) << Pmats[1](0,3), Pmats[1](1,3), Pmats[1](2,3));
    cv::Mat_<double> R = (cv::Mat_<double>(3,3) << Pmats[1](0,0), Pmats[1](0,1), Pmats[1](0,2),
            Pmats[1](1,0), Pmats[1](1,1), Pmats[1](1,2),
            Pmats[1](2,0), Pmats[1](2,1), Pmats[1](2,2));
    cv::Matx34d P_0;
    cv::Matx34d P_1;

    P_0 = cv::Matx34d(1,0,0,0,
                      0,1,0,0,
                      0,0,1,0);
    int img_prev;
    for( int img_now = 2; img_now<filelist.size(); img_now++)
    {
        cout << endl << endl << endl <<endl;
        cout << "dealing with " << filelist.at(img_now).fileName().toStdString() << endl;
        cout << endl;

        img_prev = img_now - 1;

        descriptors1.release();
        descriptors1 = descriptors2;

        QString ymlFile;
        ymlFile = ymlFileDir;
        ymlFile.append(filelist.at(img_now).fileName()).append(".yml");
        restore_descriptors_from_file(ymlFile.toStdString(),imgpts[img_now],descriptors2);

        matches_prev.clear();
        matches_prev = matches_new;
        matches_new.clear();
        //matching
        matching_fb_matcher(descriptors1,descriptors2,matches_new);
        matching_good_matching_filter(matches_new);

        outCloud_prev = outCloud_new;
        outCloud_new.clear();

        vector<cv::Point3f> tmp3d; vector<cv::Point2f> tmp2d;

        for (unsigned int i=0; i < matches_new.size(); i++) {
            int idx_in_prev_img = matches_new[i].queryIdx;
            for (unsigned int pcldp=0; pcldp<outCloud_prev.size(); pcldp++)
            {
                if(idx_in_prev_img == outCloud_prev[pcldp].imgpt_for_img[img_prev])
                {
                    tmp3d.push_back(outCloud_prev[pcldp].pt);
                    tmp2d.push_back(imgpts[img_now][matches_new[i].trainIdx].pt);
                    break;
                }
            }
        }

        bool pose_estimated = FindPoseEstimation(rvec,t,R,tmp3d,tmp2d);
        if(!pose_estimated)
        {
            cout << "error"<<endl;
        }

        //store estimated pose
        Pmats[img_now] = cv::Matx34d	(R(0,0),R(0,1),R(0,2),t(0),
                                         R(1,0),R(1,1),R(1,2),t(1),
                                         R(2,0),R(2,1),R(2,2),t(2));

        imgpts1_tmp.clear();
        imgpts2_tmp.clear();

        GetAlignedPointsFromMatch(imgpts[img_prev], imgpts[img_now], matches_new, imgpts1_tmp, imgpts2_tmp);

        std::vector<cv::KeyPoint> correspImg1Pt;
        double mean_proj_err = TriangulatePoints(imgpts1_tmp, imgpts2_tmp, K, Kinv,distcoeff, Pmats[img_prev], Pmats[img_now], outCloud, correspImg1Pt);
        std::vector<CloudPoint> outCloud_tmp;
        outCloud_tmp.clear();
        for (unsigned int i=0; i<outCloud.size(); i++)
        {
            if(outCloud[i].reprojection_error <= 5){
                //cout << "surving" << endl;
                outCloud[i].imgpt_for_img.resize(filelist.size());
                for(int j = 0; j<filelist.size();j++)
                {
                    outCloud[i].imgpt_for_img[j] = -1;
                }
                outCloud[i].imgpt_for_img[img_now] = matches_new[i].trainIdx;
                outCloud_tmp.push_back(outCloud[i]);
            }
        }
        outCloud.clear();
        outCloud= outCloud_tmp;

        for(unsigned int i=0;i<outCloud.size();i++)
        {
            outCloud_all.push_back(outCloud[i]);
        }

        outCloud_new = outCloud;
    }

    GetRGBForPointCloud(outCloud_all,pointCloudRGB);

    ui->widget->update(getPointCloud(),
                       getPointCloudRGB(),
                       getCameras());

}

/* ------------------------------------------------------------------------- */
/** \fn void on_method2_clicked()
*
* \brief method 2 for reconstruction
* only reconstruction by each two images
*/
/* ------------------------------------------------------------------------- */
void MainWindow::on_method2_clicked()
{
    on_PB_Sift_clicked();

    cout << endl << endl << endl << "Using Method 2:" <<endl;

    vector<DMatch> matches;
    cv::Matx34d P_0;
    cv::Matx34d P_1;

    P_0 = cv::Matx34d(1,0,0,0,
                      0,1,0,0,
                      0,0,1,0);

    imgpts.resize(filelist.size());

    reconstruct_first_two_view();

    cv::Mat_<double> t_prev = (cv::Mat_<double>(3,1) << 0, 0, 0);
    cv::Mat_<double> R_prev = (cv::Mat_<double>(3,3) << 0, 0, 0,
                               0, 0, 0,
                               0, 0, 0);
    cv::Mat_<double> R_prev_inv = (cv::Mat_<double>(3,3) << 0, 0, 0,
                                   0, 0, 0,
                                   0, 0, 0);
    cv::Mat_<double> t_now = (cv::Mat_<double>(3,1) << 0, 0, 0);
    cv::Mat_<double> R_now = (cv::Mat_<double>(3,3) << 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0);
    cv::Mat_<double> t_new = (cv::Mat_<double>(3,1) << 0, 0, 0);
    cv::Mat_<double> R_new = (cv::Mat_<double>(3,3) << 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0);
    int index_prev;

    std::cout << "Pmat[0]  = " << endl << Pmats[0]<<endl;
    std::cout << "Pmat[1]  = " << endl << Pmats[1]<<endl;
    for( int index_now = 2; index_now<filelist.size(); index_now++)
    {
        cout << endl << endl << endl <<endl;
        cout << "dealing with " << filelist.at(index_now).fileName().toStdString() << endl;
        cout << endl;

        index_prev = index_now - 1;
        descriptors1.release();
        descriptors1 = descriptors2;

        QString ymlFile;
        ymlFile = ymlFileDir;
        ymlFile.append(filelist.at(index_now).fileName()).append(".yml");
        restore_descriptors_from_file(ymlFile.toStdString(),imgpts[index_now],descriptors2);

        matches.clear();
        //matching
        matching_fb_matcher(descriptors1,descriptors2,matches);
        matching_good_matching_filter(matches);

        if(FindCameraMatrices(K,Kinv,distcoeff,
                              imgpts[index_prev],imgpts[index_now],
                              imggoodpts1,imggoodpts2,
                              P_0,P_1,
                              matches,
                              outCloud))
        {//if can find camera matries


            R_prev(0,0) = Pmats[index_prev](0,0);
            R_prev(0,1) = Pmats[index_prev](0,1);
            R_prev(0,2) = Pmats[index_prev](0,2);
            R_prev(1,0) = Pmats[index_prev](1,0);
            R_prev(1,1) = Pmats[index_prev](1,1);
            R_prev(1,2) = Pmats[index_prev](1,2);
            R_prev(2,0) = Pmats[index_prev](2,0);
            R_prev(2,1) = Pmats[index_prev](2,1);
            R_prev(2,2) = Pmats[index_prev](2,2);
            t_prev(0,0) = Pmats[index_prev](0,3);
            t_prev(1,0) = Pmats[index_prev](1,3);
            t_prev(2,0) = Pmats[index_prev](2,3);

            R_now(0,0) = P_1(0,0);
            R_now(0,1) = P_1(0,1);
            R_now(0,2) = P_1(0,2);
            R_now(1,0) = P_1(1,0);
            R_now(1,1) = P_1(1,1);
            R_now(1,2) = P_1(1,2);
            R_now(2,0) = P_1(2,0);
            R_now(2,1) = P_1(2,1);
            R_now(2,2) = P_1(2,2);
            t_now(0,0) = P_1(0,3);
            t_now(1,0) = P_1(1,3);
            t_now(2,0) = P_1(2,3);

            invert(R_prev, R_prev_inv);

            t_new = R_prev*t_now + t_prev;
            R_new = R_now*R_prev;
            //        //store estimated pose
            Pmats[index_now] = cv::Matx34d	(R_new(0,0),R_new(0,1),R_new(0,2),t_new(0),
                                             R_new(1,0),R_new(1,1),R_new(1,2),t_new(1),
                                             R_new(2,0),R_new(2,1),R_new(2,2),t_new(2));
            cout << "Pmats[index_now]:" << endl << Pmats[index_now]  << endl;

            //        Pmats[index_now] = cv::Matx34d	(P_0(0,0),P_0(0,1),P_0(0,2),P_1(0,3),
            //                                         P_0(1,0),P_0(1,1),P_0(1,2),P_1(1,3),
            //                                         P_0(2,0),P_0(2,1),P_0(2,2),P_1(2,3));

        }
        else
        {
            break;
        }



    }
    cout << "finished" <<endl <<endl;


}





















//pop the data to the visualization module
std::vector<cv::Point3d> MainWindow::getPointCloud()
{
    return CloudPointsToPoints(outCloud_all);
}

const std::vector<cv::Vec3b>& MainWindow::getPointCloudRGB()
{
    return pointCloudRGB;
}

std::vector<cv::Matx34d> MainWindow::getCameras()
{
    std::vector<cv::Matx34d> v;
    for(std::map<int ,cv::Matx34d>::const_iterator it = Pmats.begin(); it != Pmats.end(); ++it )
    {
        v.push_back( it->second );
    }
    return v;
}


void MainWindow::GetRGBForPointCloud(
        const std::vector<struct CloudPoint>& _pcloud,
        std::vector<cv::Vec3b>& RGBforCloud
        )
{
    RGBforCloud.resize(_pcloud.size());
    for (unsigned int i=0; i<_pcloud.size(); i++) {
        unsigned int good_view = 0;
        std::vector<cv::Vec3b> point_colors;
        for(; good_view < imgs_orig.size(); good_view++) {
            if(_pcloud[i].imgpt_for_img[good_view] != -1) {
                unsigned int pt_idx = _pcloud[i].imgpt_for_img[good_view];
                if(pt_idx >= imgpts[good_view].size()) {
                    std::cerr << i << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << std::endl;
                    continue;
                }
                cv::Point _pt = imgpts[good_view][pt_idx].pt;
                assert(good_view < imgs_orig.size() && _pt.x < imgs_orig[good_view].cols && _pt.y < imgs_orig[good_view].rows);

                point_colors.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));
                //				imshow_250x250(ss.str(), imgs_orig[good_view](cv::Range(_pt.y-10,_pt.y+10),cv::Range(_pt.x-10,_pt.x+10)));
            }
        }
        //		cv::waitKey(0);
        cv::Scalar res_color = cv::mean(point_colors);
        RGBforCloud[i] = (cv::Vec3b(res_color[0],res_color[1],res_color[2])); //bgr2rgb
        //        if(good_view == imgs.size()) //nothing found.. put red dot
        //            RGBforCloud.push_back(cv::Vec3b(255,0,0));
    }
}

void MainWindow::on_pushButton_clicked()
{
    //List the transmition of the camera.
    for( int i = 0; i<filelist.size(); i++)
        std::cout << -1*Pmats[i](2,3) << endl;
    cout << endl;
    for( int i = 0; i<filelist.size(); i++)
        std::cout << -1*Pmats[i](0,3) << endl;
    cout << endl;

    //    for( int i = 0; i<filelist.size(); i++)
    //        std::cout << Pmats[i](1,3) << endl;

}


