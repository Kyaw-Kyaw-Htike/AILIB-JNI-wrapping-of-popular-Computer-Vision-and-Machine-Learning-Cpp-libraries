#pragma once

#include "KKH_StdLib_ailib.h"

// headers: C++ standard
#include <iostream> // cin,cout,cerr
#include <vector> // a container for a dynamic array
#include <array> // since C++11: a container for a fixed sized array
#include <string> // string classes and templates
#include <random> // since C++11: Random number generators and distributions
#include <fstream> //  classes for file-based input and output
#include <sstream> // classes for string manipulation: stringstream
#include <cstdio> //  printf,fprintf,fopen,fclose,fread,sscanf,fgets,FILE
#include <cmath> // ceil,floor,fabs,abs,cos,sin,cosh,acos,asin,atan2,exp,log,pow,sqrt
#include <cstdlib> // system(“PAUSE”),malloc,calloc,free,rnd,srand,atof,atoi,strtod
#include <cstring> // memcpy,memmove,strcpy,strcat,memset,strlen,strcmp
#include <cctype> // islower,isupper,tolower,toupper,isdigit
#include <algorithm>

// headers: OpenCV
//#include "opencv2/opencv.hpp" // this includes all header files; better to do below
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"

// headers: Armadillo linear alg
#include "armadillo"

// headers: Eigen
#include "Eigen/Dense"

// headers: dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
//#include "dlib/all/source.cpp" // I won't need it usually

#include "JNI_type_converter.h"

//using namespace std; // for standard C++ lib
//using namespace cv; // for opencv
//using namespace arma; // for Armadillo linear alg lib
//using namespace Eigen; // for Eigen linear alg lib
//using namespace dlib; // for dlib lib

#include "Header_hogDollar.h"

// for vlfeat
#include "Header_vlfeat_helper.h"
#include "kmeans.h"
#include "lbp.h"
#include "hog.h"
#include "svm.h"
#include "sift.h"
#include "dsift.h"