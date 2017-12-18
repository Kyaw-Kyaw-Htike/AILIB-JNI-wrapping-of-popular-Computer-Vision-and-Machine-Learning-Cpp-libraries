// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

#include <jni.h>
/* Header for class KKH_StdLib_ailib */

#ifndef _Included_KKH_StdLib_ailib
#define _Included_KKH_StdLib_ailib
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     KKH_StdLib_ailib
 * Method:    imread
 * Signature: (Ljava/lang/String;Z)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_imread
  (JNIEnv *, jclass, jstring, jboolean);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    imwrite
 * Signature: (LKKH/StdLib/Matkc;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_imwrite
  (JNIEnv *, jclass, jobject, jstring);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    imshow
 * Signature: (LKKH/StdLib/Matkc;ILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_imshow
  (JNIEnv *, jclass, jobject, jint, jstring);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    imresize
 * Signature: (LKKH/StdLib/Matkc;IIDDI)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_imresize
  (JNIEnv *, jclass, jobject, jint, jint, jdouble, jdouble, jint);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    cvCreateFileCapture
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_KKH_StdLib_ailib_cvCreateFileCapture
  (JNIEnv *, jclass, jstring);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    cvCreateCameraCapture
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_KKH_StdLib_ailib_cvCreateCameraCapture
  (JNIEnv *, jclass, jint);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    cvReleaseCapture
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_cvReleaseCapture
  (JNIEnv *, jclass, jlong);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    cvQueryFrame
 * Signature: (J)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_cvQueryFrame
  (JNIEnv *, jclass, jlong);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    gradMO_raw
 * Signature: (LKKH/StdLib/Matkc;Z)LKKH/StdLib/ailib/Results_HogDollar_GradsRaw;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_gradMO_1raw
  (JNIEnv *, jclass, jobject, jboolean);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    gradHist_hog_fhog
 * Signature: (LKKH/StdLib/Matkc;IIIIZF)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_gradHist_1hog_1fhog
  (JNIEnv *, jclass, jobject, jint, jint, jint, jint, jboolean, jfloat);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_svm_train
 * Signature: (LKKH/StdLib/Matkc;LKKH/StdLib/Matkc;DZZ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_KKH_StdLib_ailib_vl_1svm_1train
  (JNIEnv *, jclass, jobject, jobject, jdouble, jboolean, jboolean);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_lbp_extract
 * Signature: (LKKH/StdLib/Matkc;I)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1lbp_1extract
  (JNIEnv *, jclass, jobject, jint);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_hog_extract
 * Signature: (LKKH/StdLib/Matkc;IZIZ)LKKH/StdLib/Matkc;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1hog_1extract
  (JNIEnv *, jclass, jobject, jint, jboolean, jint, jboolean);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_kmeans
 * Signature: (LKKH/StdLib/Matkc;IIIIIIIII)LKKH/StdLib/ailib/Results_vl_kmeans;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1kmeans
  (JNIEnv *, jclass, jobject, jint, jint, jint, jint, jint, jint, jint, jint, jint);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_sift
 * Signature: (LKKH/StdLib/Matkc;IIIDDDDDLKKH/StdLib/Matkc;ZI)LKKH/StdLib/ailib/Results_vl_sift;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1sift_1jni
  (JNIEnv *, jclass, jobject, jint, jint, jint, jdouble, jdouble, jdouble, jdouble, jdouble, jobject, jboolean, jint);

/*
 * Class:     KKH_StdLib_ailib
 * Method:    vl_dsift_jni
 * Signature: (LKKH/StdLib/Matkc;[I[I[DZZ[IZ)LKKH/StdLib/ailib/Results_vl_sift;
 */
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1dsift_1jni
  (JNIEnv *, jclass, jobject, jintArray, jintArray, jdoubleArray, jboolean, jboolean, jintArray, jboolean);

#ifdef __cplusplus
}
#endif
#endif
