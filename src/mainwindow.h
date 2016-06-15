#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileSystemModel>
#include <QDebug>
#include "QFileSystemModel"
#include "QDir"
#include "qdir.h"
#include <unistd.h>
#include <fcntl.h>

#include "opencv2/core/core.hpp"
#include "matching.h"
#include "findcameramatrices.h"
#include "Triangulation.h"
#include "sfmviewer.h"
#include "common.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>



typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_path_returnPressed();

    void on_PB_Sift_clicked();

    void on_PB_Reconstruction_clicked();

    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;

    std::vector<cv::Mat> imgs_orig; //color iamge vec


    //common
    QFileInfoList filelist;
    QString ymlFileDir;

    //camera parameter
    Mat K;
    cv::Mat_<double> Kinv;
    Mat distcoeff;

    //matching
    std::vector<std::vector<cv::KeyPoint> > imgpts;
    vector<KeyPoint> imggoodpts1,imggoodpts2;
    Mat descriptors1,descriptors2;
    vector<DMatch> matches_prev;
    vector<DMatch> matches_new;

    cv::Matx34d P_first;
    cv::Matx34d P_second;
    std::vector<CloudPoint> outCloud;
    std::vector<CloudPoint> outCloud_prev;
    std::vector<CloudPoint> outCloud_new;
    std::vector<CloudPoint> outCloud_all;


    //for multiple view
    std::map<int,cv::Matx34d> Pmats;

    std::vector<cv::Vec3b>  pointCloudRGB;

    void reconstruct_first_two_view();

    void GetRGBForPointCloud(
            const std::vector<struct CloudPoint>& _pcloud,
            std::vector<cv::Vec3b>& RGBforCloud
            );


    //visualization
    PointCloudT::Ptr cloud_;


    bool FindPoseEstimation(
        cv::Mat_<double>& rvec,
        cv::Mat_<double>& t,
        cv::Mat_<double>& R,
        std::vector<cv::Point3f> ppcloud,
        std::vector<cv::Point2f> imgPoints);


    //pop the data to the visualization module
    std::vector<cv::Point3d> getPointCloud();

    const std::vector<cv::Vec3b>& getPointCloudRGB();

    std::vector<cv::Matx34d> getCameras();

};

#endif // MAINWINDOW_H
