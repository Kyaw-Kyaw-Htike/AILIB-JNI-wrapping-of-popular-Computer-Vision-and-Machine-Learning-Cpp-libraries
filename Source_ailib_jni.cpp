#include "Header_ailib.h"
#include <fstream>

// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
// Only the rights corresponding to the wrapping code belongs to the author. The portions of code from the OpenCV, Piotr Dollar and VLFeat libraries belong to the respective owners.

/*
* Class:     KKH_StdLib_ailib
* Method:    imread
* Signature: (Ljava/lang/String;Z)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_imread
(JNIEnv * env, jclass cls, jstring fpath_, jboolean divBy255_)
{
	jni_utils ju(env);
	std::string fpath = ju.from_jstring(fpath_);
	bool divBy255 = ju.from_jboolean(divBy255_);
	cv::Mat img = cv::imread(fpath);
	Matkc m;
	m.create<unsigned char, 3>(env, img, divBy255);
	jArray<jdoubleArray, jdouble> mm(env);
	return m.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    imshow
* Signature: (LKKH/StdLib/Matkc;ILjava/lang/String;)V
*/
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_imshow
(JNIEnv *env, jclass cls, jobject img_, jint delay_, jstring name_win_)
{
	jni_utils ju(env);
	Matkc img; img.create(env, img_);
	cv::Mat imgcv;
	switch (img.nchannels())
	{
	case 1:
		imgcv = img.to_cvMat<unsigned char, 1>();
		break;
	case 3:
		imgcv = img.to_cvMat<unsigned char, 3>();
		break;
	default:
		ju.throw_exception("ERROR from JNI: Input image must be either 1 or 3 channels");
		return;
	}
		
	int delay = delay_;
	std::string name_win = ju.from_jstring(name_win_);

	cv::imshow(name_win, imgcv);
	cv::waitKey(delay);
}

/*
* Class:     KKH_StdLib_ailib
* Method:    imresize
* Signature: (LKKH/StdLib/Matkc;IIDDI)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_imresize
(JNIEnv *env, jclass cls, jobject img_, jint width_new_, jint height_new_, jdouble fx_, jdouble fy_, jint interpolation_)
{
	jni_utils ju(env);

	Matkc img; img.create(env, img_);
	cv::Mat imgcv;
	int nchannels = img.nchannels();
	switch (nchannels)
	{
	case 1:
		imgcv = img.to_cvMat<double, 1>();
		break;
	case 3:
		imgcv = img.to_cvMat<double, 3>();
		break;
	default:
		ju.throw_exception("ERROR from JNI: Input image must be either 1 or 3 channels");
		return nullptr;
	}

	int width_new = width_new_;
	int height_new = height_new_;
	double fx = fx_;
	double fy = fy_;
	int interpolation = interpolation_;
	cv::Size size_img_new(width_new, height_new);

	cv::Mat img_resized;
	cv::resize(imgcv, img_resized, size_img_new, fx, fy, interpolation);

	Matkc img_out; 

	switch (nchannels)
	{
	case 1:
		img_out.create<double, 1>(env, img_resized);
		break;
	case 3:
		img_out.create<double, 3>(env, img_resized);
		break;
	}

	return img_out.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    imwrite
* Signature: (LKKH/StdLib/Matkc;Ljava/lang/String;)V
*/
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_imwrite
(JNIEnv *env, jclass cls, jobject img_, jstring fpath_)
{
	jni_utils ju(env);
	Matkc img; img.create(env, img_);
	cv::Mat imgcv;
	int nchannels = img.nchannels();
	switch (nchannels)
	{
	case 1:
		imgcv = img.to_cvMat<unsigned char, 1>();
		break;
	case 3:
		imgcv = img.to_cvMat<unsigned char, 3>();
		break;
	default:
		ju.throw_exception("ERROR from JNI: Input image must be either 1 or 3 channels");
		return;
	}

	std::string fpath = ju.from_jstring(fpath_);
	cv::imwrite(fpath, imgcv);
}

/*
* Class:     KKH_StdLib_ailib
* Method:    cvCreateFileCapture
* Signature: (Ljava/lang/String;)J
*/
JNIEXPORT jlong JNICALL Java_KKH_StdLib_ailib_cvCreateFileCapture
(JNIEnv *env, jclass cls, jstring fpath_)
{
	jni_utils ju(env);
	std::string fpath = ju.from_jstring(fpath_);
	CvCapture* capObj = cvCreateFileCapture(fpath.c_str());
	if (capObj == NULL)
		ju.throw_exception("ERROR from JNI: video capture failed.");
	return (jlong)capObj;
}

/*
* Class:     KKH_StdLib_ailib
* Method:    cvCreateCameraCapture
* Signature: (I)J
*/
JNIEXPORT jlong JNICALL Java_KKH_StdLib_ailib_cvCreateCameraCapture
(JNIEnv *env, jclass cls, jint index_)
{
	jni_utils ju(env);
	CvCapture* capObj = cvCreateCameraCapture(index_);
	if (capObj == NULL)
		ju.throw_exception("ERROR from JNI: video capture failed.");
	return (jlong)capObj;
}

/*
* Class:     KKH_StdLib_ailib
* Method:    cvReleaseCapture
* Signature: (J)V
*/
JNIEXPORT void JNICALL Java_KKH_StdLib_ailib_cvReleaseCapture
(JNIEnv *env, jclass cls, jlong obj_CvCapture_)
{
	CvCapture* capObj = (CvCapture*)obj_CvCapture_;
	cvReleaseCapture(&capObj);
}

/*
* Class:     KKH_StdLib_ailib
* Method:    cvQueryFrame
* Signature: (J)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_cvQueryFrame
(JNIEnv *env, jclass cls, jlong obj_CvCapture_)
{
	CvCapture* capObj = (CvCapture*)obj_CvCapture_;
	IplImage* img_ipl = cvQueryFrame(capObj);	
	Matkc img; img.create<unsigned char, 3>(env, img_ipl);	
	return img.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    gradMO_raw
* Signature: (LKKH/StdLib/Matkc;Z)LKKH/StdLib/ailib/Results_HogDollar_GradsRaw;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_gradMO_1raw
(JNIEnv *env, jclass cls, jobject img_, jboolean full_angle_)
{
	jni_utils ju(env);
	Matkc img;
	img.create(env, img_);
	
	bool full_angle = ju.from_jboolean(full_angle_);
	
	int nr = img.nrows(); 
	int nc = img.ncols();
	int nch = img.nchannels();

	std::vector<float> imgVec = img.to_stdVec<float>();
	std::vector<float> M(nr * nc);
	std::vector<float> O(nr * nc);
	gradMag(imgVec.data(), M.data(), O.data(), nr, nc, nch, full_angle);
	
	Matkc M_mat, O_mat;
	M_mat.create(env, M, nr, nc, 1);
	O_mat.create(env, O, nr, nc, 1);

	JavaClass jc(env, "KKH/StdLib/ailib$Results_HogDollar_GradsRaw");
	jc.construct_new(M_mat.get_obj(), O_mat.get_obj());
	return jc.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    gradHist_hog_fhog
* Signature: (LKKH/StdLib/Matkc;IIIIZF)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_gradHist_1hog_1fhog
(JNIEnv *env, jclass cls, jobject img_, jint useHog_, jint binSize_, jint nOrients_, 
	jint softBin_, jboolean full_angle_, jfloat clipHog_)
{
	jni_utils ju(env);
	bool full_angle = ju.from_jboolean(full_angle_);
	int binSize = ju.from_jint(binSize_);
	int useHog = ju.from_jint(useHog_);
	int nOrients = ju.from_jint(nOrients_);
	int softBin = ju.from_jint(softBin_);
	float clipHog = ju.from_jfloat(clipHog_);

	Matkc img;
	img.create(env, img_);

	int nr = img.nrows();
	int nc = img.ncols();
	int nch = img.nchannels();	

	std::vector<float> imgVec = img.to_stdVec<float>();
	std::vector<float> M(nr * nc);
	std::vector<float> O(nr * nc);

	gradMag(imgVec.data(), M.data(), O.data(), nr, nc, nch, full_angle);

	int nr_H = nr / binSize;
	int nc_H = nc / binSize;
	int nch_H = useHog == 0 ? nOrients : (useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
	std::vector<float> H(nr_H * nc_H * nch_H);	

	switch (useHog)
	{
	case 0:
		gradHist(M.data(), O.data(), H.data(), nr, nc, binSize, nOrients, softBin, full_angle);
		break;
	case 1:
		hog(M.data(), O.data(), H.data(), nr, nc, binSize, nOrients, softBin, full_angle, clipHog);
		break;
	default:
		fhog(M.data(), O.data(), H.data(), nr, nc, binSize, nOrients, softBin, clipHog);
	}

	Matkc H_mat;
	H_mat.create(env, H, nr_H, nc_H, nch_H);
	return H_mat.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_svm_train
* Signature: (LKKH/StdLib/Matkc;LKKH/StdLib/Matkc;DZZ)[D
*/
JNIEXPORT jdoubleArray JNICALL Java_KKH_StdLib_ailib_vl_1svm_1train
(JNIEnv *env, jclass cls, jobject train_data_, jobject labels_, jdouble lambda_, jboolean SGD_, jboolean weight_bal_classes_)
{
	jni_utils ju(env);
	double lambda = ju.from_jdouble(lambda_);
	bool SGD = ju.from_jboolean(SGD_);
	bool weight_bal_classes = ju.from_jboolean(weight_bal_classes_);

	Matkc train_data;
	train_data.create(env, train_data_);

	Matkc labels;
	labels.create(env, labels_);

	VlSvmSolverType solver_type;
	if (SGD)
		solver_type = VlSvmSolverType::VlSvmSolverSgd;
	else
		solver_type = VlSvmSolverType::VlSvmSolverSdca;

	int ndims_feat = train_data.nrows();
	int ndata = train_data.ncols();

	std::vector<double> train_data_vec = train_data.to_stdVec<double>();
	std::vector<double> labels_vec = labels.to_stdVec<double>();
	
	VlSvm *svm = vl_svm_new(solver_type, train_data_vec.data(), ndims_feat, ndata, labels_vec.data(), lambda);

	// Note: these weights are not importance weights. Instead, it is
	// the other way round: they are loss weights.
	std::vector<double> weights_data(ndata);

	if (weight_bal_classes)
	{
		unsigned int ndata_pos = 0;
		unsigned int ndata_neg = 0;

		for (size_t i = 0; i < ndata; i++)
		{
			if (static_cast<int>(labels_vec[i]) == 1)
				ndata_pos++;
			else
				ndata_neg++;
		}

		std::cout << "ndata_pos = " << ndata_pos << "; ndata_neg = " << ndata_neg << std::endl;

		for (size_t i = 0; i < ndata; i++)
		{
			if (static_cast<int>(labels_vec[i]) == 1)
				weights_data[i] = static_cast<double>(ndata_neg) / ndata;
			else
				weights_data[i] = static_cast<double>(ndata_pos) / ndata;
		}
	}

	else
	{
		std::fill(weights_data.begin(), weights_data.end(), 1.0);
	}

	vl_svm_set_weights(svm, weights_data.data());

	vl_svm_train(svm);
	const double * model = vl_svm_get_model(svm);
	double bias = vl_svm_get_bias(svm);

	std::vector<double> lin_model(ndims_feat + 1);
	std::copy(model, model + ndims_feat, lin_model.begin());
	lin_model[ndims_feat] = bias;

	vl_svm_delete(svm);

	return ju.to_jdoubleArray(lin_model);
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_lbp_extract
* Signature: (LKKH/StdLib/Matkc;I)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1lbp_1extract
(JNIEnv *env, jclass cls, jobject img_, jint cellSize_)
{
	jni_utils ju(env);
	int cellSize = ju.from_jint(cellSize_);
	Matkc img; img.create(env, img_);

	VlLbp * lbp = vl_lbp_new(VlLbpUniform, VL_TRUE);
	if (lbp == nullptr)
	{
		ju.throw_exception("ERROR from JNI: Could not create LBP object.\n");
		return NULL;
	}
	
	int nrows = img.nrows();
	int ncols = img.ncols();

	int nrows_H = nrows / cellSize;
	int ncols_H = ncols / cellSize;
	int nchannels_H = vl_lbp_get_dimension(lbp);
	
	std::vector<float> H(nrows_H * ncols_H * nchannels_H);
	std::vector<float> img_vec = img.to_stdVec<float>();
	H.resize(nrows_H * ncols_H * nchannels_H);
	vl_lbp_process(lbp, H.data(), img_vec.data(), nrows, ncols, cellSize);	
	vl_lbp_delete(lbp);

	Matkc H_out;
	H_out.create(env, H, nrows_H, ncols_H, nchannels_H);
	return H_out.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_hog_extract
* Signature: (LKKH/StdLib/Matkc;IZIZ)LKKH/StdLib/Matkc;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1hog_1extract
(JNIEnv *env, jclass cls, jobject img_, jint cellSize_, jboolean dalal_hog_, jint numOrientations_, jboolean bilinearOrientations_)
{
	jni_utils ju(env);
	int cellSize = ju.from_jint(cellSize_);
	bool dalal_hog = ju.from_jboolean(dalal_hog_);
	int numOrientations = ju.from_jint(numOrientations_);
	bool bilinearOrientations = ju.from_jboolean(bilinearOrientations_);
	Matkc img; img.create(env, img_);

	int width_img = img.ncols();
	int height_img = img.nrows();
	int nchannels_img = img.nchannels();

	if ((nchannels_img != 1) && (nchannels_img != 3))
	{
		ju.throw_exception("ERROR from JNI: input image must be either 1 or 3 channels.\n");
		return NULL;
	}

	VlHogVariant hog_type = VlHogVariantUoctti;
	if (dalal_hog)
		hog_type = VlHogVariantDalalTriggs;

	std::vector<float> img_vec = img.to_stdVec<float>();
	
	VlHog* hog = vl_hog_new(hog_type, numOrientations, VL_TRUE);
	vl_hog_set_use_bilinear_orientation_assignments(hog, (bilinearOrientations ? VL_TRUE : VL_FALSE));
	vl_hog_put_image(hog, img_vec.data(), height_img, width_img, nchannels_img, cellSize);

	int hogWidth = vl_hog_get_width(hog);
	int hogHeight = vl_hog_get_height(hog);
	int hogDimension = vl_hog_get_dimension(hog);
	std::vector<float> feats_vec(hogWidth * hogHeight * hogDimension);

	vl_hog_extract(hog, feats_vec.data());
	vl_hog_delete(hog);

	Matkc H;
	H.create(env, feats_vec, hogWidth, hogHeight, hogDimension);	
	return H.get_obj();
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_kmeans
* Signature: (LKKH/StdLib/Matkc;IIIIIIIII)LKKH/StdLib/ailib/Results_vl_kmeans;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1kmeans
(JNIEnv *env, jclass cls, jobject train_data_, jint nclusters_, jint alg_, jint dist_type_, jint maxIters_, jint num_repeats_, jint init_method_, jint verbosity_, jint num_comparisons_, jint ntrees_)
{
	jni_utils ju(env);
	Matkc train_data; train_data.create(env, train_data_);
	int nclusters = ju.from_jint(nclusters_);
	int alg = ju.from_jint(alg_);
	int dist_type = ju.from_jint(dist_type_);
	int maxIters = ju.from_jint(maxIters_);
	int num_repeats = ju.from_jint(num_repeats_);
	int init_method = ju.from_jint(init_method_);
	int verbosity = ju.from_jint(verbosity_);
	int num_comparisons = ju.from_jint(num_comparisons_);
	int ntrees = ju.from_jint(ntrees_);

	vl_size dimension = train_data.nrows();
	vl_size numData = train_data.ncols();
	vl_size numCenters = nclusters;

	VlKMeansAlgorithm algorithm = static_cast<VlKMeansAlgorithm>(alg);
	VlVectorComparisonType distance = static_cast<VlVectorComparisonType>(dist_type);
	vl_size maxNumIterations = maxIters;
	vl_size numRepetitions = num_repeats;
	double minEnergyVariation = -1;
	double energy;
	VlKMeansInitialization initialization = static_cast<VlKMeansInitialization>(init_method);
	vl_size maxNumComparisons = num_comparisons;
	vl_size numTrees = ntrees;
	vl_type dataType = VL_TYPE_DOUBLE;

	VlKMeans * kmeans;

	vl_set_printf_func(printf);
	vl_set_alloc_func(malloc, realloc, calloc, free);

	kmeans = vl_kmeans_new(dataType, distance);

	vl_kmeans_set_verbosity(kmeans, verbosity);
	vl_kmeans_set_num_repetitions(kmeans, numRepetitions);
	vl_kmeans_set_algorithm(kmeans, algorithm);
	vl_kmeans_set_initialization(kmeans, initialization);
	vl_kmeans_set_max_num_iterations(kmeans, maxNumIterations);
	vl_kmeans_set_max_num_comparisons(kmeans, maxNumComparisons);
	vl_kmeans_set_num_trees(kmeans, numTrees);
	if (minEnergyVariation >= 0) {
		vl_kmeans_set_min_energy_variation(kmeans, minEnergyVariation);
	}

	std::vector<double> train_data_vec = train_data.to_stdVec<double>();
	energy = vl_kmeans_cluster(kmeans, train_data_vec.data(), dimension, numData, numCenters);

	Matkc centers;
	centers.create(env, (double*)vl_kmeans_get_centers(kmeans), dimension, vl_kmeans_get_num_centers(kmeans),1);
		
	std::vector<vl_uint32> labels_vec(numData);

	vl_kmeans_quantize(kmeans, labels_vec.data(), NULL, train_data_vec.data(), numData);
	vl_kmeans_delete(kmeans);

	// make cluster labels range from 1 to K instead of 0 to K-1
	for (int i = 0; i < labels_vec.size(); i++)
		labels_vec[i]++;

	Matkc labels;
	labels.create(env, labels_vec, 1, numData, 1);

	JavaClass jc(env, "KKH/StdLib/ailib$Results_vl_kmeans");
	jc.construct_new(centers.get_obj(), labels.get_obj());
	return jc.get_obj();	
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_sift
* Signature: (LKKH/StdLib/Matkc;IIIDDDDDLKKH/StdLib/Matkc;ZI)LKKH/StdLib/ailib/Results_vl_sift;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1sift_1jni
(JNIEnv *env, jclass cls, jobject img_, jint octaves_, jint levels_, jint firstOctave_, jdouble peakThresh_, jdouble edgeThresh_, jdouble normThresh_, jdouble magnif_, jdouble windowSize_, jobject framesMat_, jboolean computeOrientation_, jint verbose_)
{
	jni_utils ju(env);
	
	Matkc img; img.create(env, img_);
	std::vector<float> img_vec = img.to_stdVec<float>();	
	Matkc framesMat; framesMat.create(env, framesMat_);
	std::vector<double> frames_vec = framesMat.to_stdVec<double>();

	float* data = img_vec.data();
	double* ikeys = frames_vec.data();
	int nikeys = -1;

	int M = img.nrows();
	int N = img.ncols();

	int O = ju.from_jint(octaves_);
	int S = ju.from_jint(levels_);
	int o_min = ju.from_jint(firstOctave_);
	double peak_thresh = ju.from_jdouble(peakThresh_);
	double edge_thresh = ju.from_jdouble(edgeThresh_);
	double norm_thresh = ju.from_jdouble(normThresh_);
	double magnif = ju.from_jdouble(magnif_);
	double window_size = ju.from_jdouble(windowSize_);
	int verbose = ju.from_jint(verbose_);
	bool computeOrientation = ju.from_jboolean(computeOrientation_);

	vl_bool force_orientations = 0;
	vl_bool floatDescriptors = 1;

	if (computeOrientation) force_orientations = 1;
	
	nikeys = framesMat.ncols();
	if (nikeys == 0) 
		nikeys = -1;
	else
	{
		if (!check_sorted(ikeys, nikeys)) {
			qsort(ikeys, nikeys, 4 * sizeof(double), korder);
		}
	}
		
	///* -----------------------------------------------------------------
	//*                                                            Do job
	//* -------------------------------------------------------------- */
	VlSiftFilt        *filt;
	vl_bool            first;
	double            *frames = 0;
	void              *descr = 0;
	int                nframes = 0, reserved = 0, i, j, q;

	/* create a filter to process the image */
	filt = vl_sift_new(M, N, O, S, o_min);

	if (peak_thresh >= 0) vl_sift_set_peak_thresh(filt, peak_thresh);
	if (edge_thresh >= 0) vl_sift_set_edge_thresh(filt, edge_thresh);
	if (norm_thresh >= 0) vl_sift_set_norm_thresh(filt, norm_thresh);
	if (magnif >= 0) vl_sift_set_magnif(filt, magnif);
	if (window_size >= 0) vl_sift_set_window_size(filt, window_size);

	if (verbose) {
		printf("vl_sift: filter settings:\n");
		printf("vl_sift:   octaves      (O)      = %d\n",
			vl_sift_get_noctaves(filt));
		printf("vl_sift:   levels       (S)      = %d\n",
			vl_sift_get_nlevels(filt));
		printf("vl_sift:   first octave (o_min)  = %d\n",
			vl_sift_get_octave_first(filt));
		printf("vl_sift:   edge thresh           = %g\n",
			vl_sift_get_edge_thresh(filt));
		printf("vl_sift:   peak thresh           = %g\n",
			vl_sift_get_peak_thresh(filt));
		printf("vl_sift:   norm thresh           = %g\n",
			vl_sift_get_norm_thresh(filt));
		printf("vl_sift:   window size           = %g\n",
			vl_sift_get_window_size(filt));
		printf("vl_sift:   float descriptor      = %d\n",
			floatDescriptors);

		printf((nikeys >= 0) ?
			"vl_sift: will source frames? yes (%d read)\n" :
			"vl_sift: will source frames? no\n", nikeys);
		printf("vl_sift: will force orientations? %s\n",
			force_orientations ? "yes" : "no");
	}

	///* ...............................................................
	//*                                             Process each octave
	//* ............................................................ */
	i = 0;
	first = 1;
	while (1) {
		int                   err;
		VlSiftKeypoint const *keys = 0;
		int                   nkeys = 0;

		if (verbose) {
			printf("vl_sift: processing octave %d\n",
				vl_sift_get_octave_index(filt));
		}

		/* Calculate the GSS for the next octave .................... */
		if (first) {
			err = vl_sift_process_first_octave(filt, data);
			first = 0;
		}
		else {
			err = vl_sift_process_next_octave(filt);
		}

		if (err) break;

		if (verbose > 1) {
			printf("vl_sift: GSS octave %d computed\n",
				vl_sift_get_octave_index(filt));
		}

		/* Run detector ............................................. */
		if (nikeys < 0) {
			vl_sift_detect(filt);

			keys = vl_sift_get_keypoints(filt);
			nkeys = vl_sift_get_nkeypoints(filt);
			i = 0;

			if (verbose > 1) {
				printf("vl_sift: detected %d (unoriented) keypoints\n", nkeys);
			}
		}
		else {
			nkeys = nikeys;
		}

		/* For each keypoint ........................................ */
		for (; i < nkeys; ++i) {
			double                angles[4];
			int                   nangles;
			VlSiftKeypoint        ik;
			VlSiftKeypoint const *k;

			/* Obtain keypoint orientations ........................... */
			if (nikeys >= 0) {
				vl_sift_keypoint_init(filt, &ik,
					ikeys[4 * i + 1] - 1,
					ikeys[4 * i + 0] - 1,
					ikeys[4 * i + 2]);

				if (ik.o != vl_sift_get_octave_index(filt)) {
					break;
				}

				k = &ik;

				/* optionally compute orientations too */
				if (force_orientations) {
					nangles = vl_sift_calc_keypoint_orientations
					(filt, angles, k);
				}
				else {
					angles[0] = VL_PI / 2 - ikeys[4 * i + 3];
					nangles = 1;
				}
			}
			else {
				k = keys + i;
				nangles = vl_sift_calc_keypoint_orientations
				(filt, angles, k);
			}

			/* For each orientation ................................... */
			for (q = 0; q < nangles; ++q) {
				vl_sift_pix  buf[128];
				vl_sift_pix rbuf[128];

				/* compute descriptor */
				vl_sift_calc_keypoint_descriptor(filt, buf, k, angles[q]);
				transpose_descriptor(rbuf, buf);

				/* make enough room for all these keypoints and more */
				if (reserved < nframes + 1) {
					reserved += 2 * nkeys;
					frames = (double*)realloc(frames, 4 * sizeof(double) * reserved);			
					if (!floatDescriptors) {
						descr = realloc(descr, 128 * sizeof(vl_uint8) * reserved);
					}
					else {
						descr = realloc(descr, 128 * sizeof(float) * reserved);
					}
				}

				/* Save back with MATLAB conventions. Notice tha the input
				* image was the transpose of the actual image. */
				frames[4 * nframes + 0] = k->y + 1;
				frames[4 * nframes + 1] = k->x + 1;
				frames[4 * nframes + 2] = k->sigma;
				frames[4 * nframes + 3] = VL_PI / 2 - angles[q];
		
				if (!floatDescriptors) {
					for (j = 0; j < 128; ++j) {
						float x = 512.0F * rbuf[j];
						x = (x < 255.0F) ? x : 255.0F;
						((vl_uint8*)descr)[128 * nframes + j] = (vl_uint8)x;
					}
				}
				else {
					for (j = 0; j < 128; ++j) {
						float x = 512.0F * rbuf[j];
						((float*)descr)[128 * nframes + j] = x;
					}
				}

				++nframes;
			} /* next orientation */
		} /* next keypoint */
	} /* next octave */

	if (verbose)
		printf("vl_sift: found %d keypoints\n", nframes);
		
	///* ...............................................................
	//*                                                       Save back
	//* ............................................................ */

	Matkc frames_out;
	frames_out.create(env, frames, 4, nframes, 1);

	Matkc descriptors_out;
	descriptors_out.create(env, (float*)descr, 128, nframes, 1);

	/* cleanup */
	vl_sift_delete(filt);

	JavaClass jc(env, "KKH/StdLib/ailib$Results_vl_sift");
	jc.construct_new(descriptors_out.get_obj(), frames_out.get_obj());
	return jc.get_obj();	
}

/*
* Class:     KKH_StdLib_ailib
* Method:    vl_dsift_jni
* Signature: (LKKH/StdLib/Matkc;[I[I[DZZ[IZ)LKKH/StdLib/ailib/Results_vl_sift;
*/
JNIEXPORT jobject JNICALL Java_KKH_StdLib_ailib_vl_1dsift_1jni
(JNIEnv *env, jclass cls, jobject img_, jintArray step_, jintArray size_, jdoubleArray bounds_, jboolean output_norm_, jboolean fast_, jintArray geometry_, jboolean verbose_)
{
	jni_utils ju(env);

	Matkc img; img.create(env, img_);
	std::vector<float> img_vec = img.to_stdVec<float>();
	float* data = img_vec.data();

	int M = img.nrows();
	int N = img.ncols();

	jArray<jintArray, jint> step__(env); step__.wrap(step_);
	jArray<jintArray, jint> size__(env); size__.wrap(size_);
	jArray<jdoubleArray, jdouble> bounds__(env); bounds__.wrap(bounds_);
	bool output_norm__ = ju.from_jboolean(output_norm_);
	bool fast__ = ju.from_jboolean(fast_);
	jArray<jintArray, jint>  geometry__(env); geometry__.wrap(geometry_); 
	jboolean verbose__ = ju.from_jboolean(verbose_);

	int verbose = 0;
	int opt;
	int step[2] = { 1,1 };
	vl_bool norm = 0;
	vl_bool floatDescriptors = VL_FALSE;
	vl_bool useFlatWindow = VL_FALSE;
	double windowSize = -1.0;
	double *bounds = NULL;
	double boundBuffer[4];
	VlDsiftDescriptorGeometry geom;

	geom.numBinX = 4;
	geom.numBinY = 4;
	geom.numBinT = 8;
	geom.binSizeX = 3;
	geom.binSizeY = 3;

	if(verbose__) verbose++;
	if(fast__) useFlatWindow = 1;
	if (output_norm__) norm = 1;

	boundBuffer[0] = ju.from_jdouble(bounds__.get_val(0)) - 1;
	boundBuffer[1] = ju.from_jdouble(bounds__.get_val(1)) - 1;
	boundBuffer[2] = ju.from_jdouble(bounds__.get_val(2)) - 1;
	boundBuffer[3] = ju.from_jdouble(bounds__.get_val(3)) - 1;

	if (!(boundBuffer[0] == -2 && boundBuffer[1] == -2 && boundBuffer[2] == -2 && boundBuffer[3] == -2))
	{
		printf("from JNI: Bounds specified.\n");
		bounds = boundBuffer;
		printf("From JNI: bounds[0] = %f, bounds[1] = %f, bounds[2] = %f, bounds[3] = %f.\n", bounds[0], bounds[1], bounds[2], bounds[3]);
	}
	else
	{
		printf("from JNI: Bounds whole image desired.\n");		
	}
			
	geom.binSizeX = ju.from_jint(size__.get_val(0));
	geom.binSizeY = ju.from_jint(size__.get_val(1));

	step[0] = ju.from_jint(step__.get_val(0));
	step[1] = ju.from_jint(step__.get_val(1));

	printf("From JNI: step[0] = %d, step[1] = %d.\n", step[0], step[1]);
	printf("From JNI: size[0] = %d, size[1] = %d.\n", geom.binSizeX, geom.binSizeY);

	floatDescriptors = VL_TRUE;

	geom.numBinY = ju.from_jint(geometry__.get_val(0));
	geom.numBinX = ju.from_jint(geometry__.get_val(1));
	geom.numBinT = ju.from_jint(geometry__.get_val(2));
	
	/* -----------------------------------------------------------------
	*                                                            Do job
	* -------------------------------------------------------------- */

	int numFrames;
	int descrSize;
	VlDsiftKeypoint const *frames;
	float const *descrs;
	int k, i;

	VlDsiftFilter *dsift;

	/* note that the image received from MATLAB is transposed */
	dsift = vl_dsift_new(M, N);
	vl_dsift_set_geometry(dsift, &geom);
	vl_dsift_set_steps(dsift, step[0], step[1]);

	if (bounds) {
		vl_dsift_set_bounds(dsift,
			VL_MAX(bounds[1], 0),
			VL_MAX(bounds[0], 0),
			VL_MIN(bounds[3], M - 1),
			VL_MIN(bounds[2], N - 1));
	}

	vl_dsift_set_flat_window(dsift, useFlatWindow);

	if (windowSize >= 0) {
		vl_dsift_set_window_size(dsift, windowSize);
	}

	numFrames = vl_dsift_get_keypoint_num(dsift);
	descrSize = vl_dsift_get_descriptor_size(dsift);
	geom = *vl_dsift_get_geometry(dsift);

	if (verbose) {
		int stepX;
		int stepY;
		int minX;
		int minY;
		int maxX;
		int maxY;
		vl_bool useFlatWindow;

		vl_dsift_get_steps(dsift, &stepY, &stepX);
		vl_dsift_get_bounds(dsift, &minY, &minX, &maxY, &maxX);
		useFlatWindow = vl_dsift_get_flat_window(dsift);

		printf("vl_dsift: image size         [W, H] = [%d, %d]\n", N, M);
		printf("vl_dsift: bounds:            [minX,minY,maxX,maxY] = [%d, %d, %d, %d]\n",
			minX + 1, minY + 1, maxX + 1, maxY + 1);
		printf("vl_dsift: subsampling steps: stepX=%d, stepY=%d\n", stepX, stepY);
		printf("vl_dsift: num bins:          [numBinT, numBinX, numBinY] = [%d, %d, %d]\n",
			geom.numBinT,
			geom.numBinX,
			geom.numBinY);
		printf("vl_dsift: descriptor size:   %d\n", descrSize);
		printf("vl_dsift: bin sizes:         [binSizeX, binSizeY] = [%d, %d]\n",
			geom.binSizeX,
			geom.binSizeY);
		printf("vl_dsift: flat window:       %s\n", VL_YESNO(useFlatWindow));
		printf("vl_dsift: window size:       %g\n", vl_dsift_get_window_size(dsift));
		printf("vl_dsift: num of features:   %d\n", numFrames);
	}

	vl_dsift_process(dsift, data);

	frames = vl_dsift_get_keypoints(dsift);
	descrs = vl_dsift_get_descriptors(dsift);

	/* ---------------------------------------------------------------
	*                                            Create output arrays
	* ------------------------------------------------------------ */
	
	int ndims_frames = norm ? 3 : 2;
	std::vector<float> descriptors_out_vec(descrSize * numFrames);
	std::vector<double> frames_out_vec(ndims_frames * numFrames);


	/* ---------------------------------------------------------------
	*                                                       Copy back
	* ------------------------------------------------------------ */

	float *tmpDescr = (float*)malloc(sizeof(float) * descrSize);
	double *outFrameIter = frames_out_vec.data();
	void *outDescrIter = descriptors_out_vec.data();
	for (k = 0; k < numFrames; ++k) {
		*outFrameIter++ = frames[k].y + 1;
		*outFrameIter++ = frames[k].x + 1;

		/* We have an implied / 2 in the norm, because of the clipping
		below */
		if (norm)
			*outFrameIter++ = frames[k].norm;

		vl_dsift_transpose_descriptor(tmpDescr,
			descrs + descrSize * k,
			geom.numBinT,
			geom.numBinX,
			geom.numBinY);

		for (i = 0; i < descrSize; ++i) {
			float * pt = (float*)outDescrIter;
			*pt++ = VL_MIN(512.0F * tmpDescr[i], 255.0F);
			outDescrIter = pt;
		}
	}
	free(tmpDescr);
	vl_dsift_delete(dsift);
	
	Matkc descriptors_out;
	descriptors_out.create(env, descriptors_out_vec, descrSize, numFrames, 1);

	Matkc frames_out;
	frames_out.create(env, frames_out_vec, ndims_frames, numFrames, 1);

	JavaClass jc(env, "KKH/StdLib/ailib$Results_vl_sift");
	jc.construct_new(descriptors_out.get_obj(), frames_out.get_obj());
	return jc.get_obj();	
}

