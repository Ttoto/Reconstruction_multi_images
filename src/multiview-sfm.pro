QT       += core gui opengl xml widgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = multiview-sfm
TEMPLATE = app


SOURCES +=  main.cpp                \
            mainwindow.cpp          \
            matching.cpp            \
            common.cpp              \
            findcameramatrices.cpp  \
            Triangulation.cpp       \
            sfmviewer.cpp           \

HEADERS  += mainwindow.h            \
            matching.h              \
            common.h                \
            findcameramatrices.h    \
            Triangulation.h         \
            sfmviewer.h             \

FORMS    += mainwindow.ui

INCLUDEPATH += "/usr/local/include/"
CONFIG   += link_pkgconfig
PKGCONFIG+= opencv



INCLUDEPATH +=  "/usr/include/pcl-1.7/"        \
                "/usr/include/pcl-1.7/pcl"     \
                "/usr/include/flann/"          \
                "/usr/include/eigen3/"         \

INCLUDEPATH += /home/sy/lib/libQGLViewer-2.6.3
LIBS += -L/home/sy/lib/libQGLViewer-2.6.3/QGLViewer
LIBS += -lQGLViewer
LIBS += -lglut

LIBS += -lpcl_common            \
        -lpcl_features          \
        -lpcl_filters           \
        -lpcl_io                \
        -lpcl_io_ply            \
        -lpcl_kdtree            \
        -lpcl_keypoints         \
        -lpcl_octree            \
        -lpcl_outofcore         \
        -lpcl_features          \
        -lflann                 \
        -lqhull                 \
        -lboost_system          \

#        -lpcl_segmentation      \
#        -lpcl_people            \
#        -lpcl_recognition       \
#        -lpcl_registration      \
#        -lpcl_sample_consensus  \
#        -lpcl_search            \
#        -lpcl_surface           \
#        -lpcl_tracking          \
#        -lpcl_visualization     \

#        -lQVTK \
#        -lvtkCommon \
#        -lvtkFiltering \
#        -lvtkRendering \
