// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package KKH.StdLib;

import java.net.InterfaceAddress;

public final class ailib {

    /**
     * The interpolation method used for image resizing
     * NN: a nearest-neighbor interpolation
     * LINEAR: a bilinear interpolation (used by opencv by default)
     * CUBIC: a bicubic interpolation over 4x4 pixel neighborhood
     * AREA:  resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
     * LANCZOS4: a Lanczos interpolation over 8x8 pixel neighborhood
     */
    public static enum Imresize_interpolation { NN, LINEAR, CUBIC, AREA, LANCZOS4 }

    public static Matkc imread(String fpath)
    {
        return imread(fpath, false);
    }

    public static native Matkc imread(String fpath, boolean divBy255);
    public static native void imwrite(Matkc img, String fpath);
    public static native void imshow(Matkc img, int delay, String name_win);
    public static void imshow(Matkc img, int delay)
    {
        imshow(img, delay, "Window opencv");
    }
    public static void imshow(Matkc img)
    {
        imshow(img, 0, "Window opencv");
    }

    public static Matkc imresize(Matkc img, int width_new, int height_new, double fx, double fy, Imresize_interpolation interpolation)
    {
        return imresize(img, width_new, height_new, fx, fy, interpolation.ordinal());
    }

    public static Matkc imresize(Matkc img, int width_new, int height_new, double fx, double fy)
    {
        return imresize(img, width_new, height_new, fx, fy, Imresize_interpolation.LINEAR);
    }

    public static Matkc imresize(Matkc img, int width_new, int height_new, Imresize_interpolation interpolation)
    {
        return imresize(img, width_new, height_new, 0, 0, interpolation);
    }

    public static Matkc imresize(Matkc img, int width_new, int height_new)
    {
        return imresize(img, width_new, height_new, 0, 0, Imresize_interpolation.LINEAR);
    }

    public static Matkc imresize(Matkc img, double scale, Imresize_interpolation interpolation)
    {
        return imresize(img, 0, 0, scale, scale, interpolation);
    }

    public static Matkc imresize(Matkc img, double scale)
    {
        return imresize(img, 0, 0, scale, scale, Imresize_interpolation.LINEAR);
    }

    private static native Matkc imresize(Matkc img, int width_new, int height_new, double fx, double fy, int interpolation);

    public static class CvVideoCapture
    {
        private long obj_CvCapture;
        public CvVideoCapture(String fpath)
        {
            obj_CvCapture = cvCreateFileCapture(fpath);
        }
        public CvVideoCapture(int index)
        {
            obj_CvCapture = cvCreateCameraCapture(index);
        }
        public void close()
        {
            cvReleaseCapture(obj_CvCapture);
        }
        public Matkc get_frame()
        {
            return cvQueryFrame(obj_CvCapture);
        }
    }

    private static native long cvCreateFileCapture(String fpath);
    private static native long cvCreateCameraCapture(int index);
    private static native void cvReleaseCapture(long obj_CvCapture);
    private static native Matkc cvQueryFrame(long obj_CvCapture);

    public static class Results_HogDollar_GradsRaw
    {
        public Matkc M;
        public Matkc O;

        public Results_HogDollar_GradsRaw(Matkc M, Matkc O)
        {
            this.M = M; this.O = O;
        }
    }

    public static class HogDollar {

        private int binSize;
        private int nOrients;
        private int softBin;
        private int useHog;
        private float clipHog;
        private boolean full_angle;

        public HogDollar()
        {
            set_params_dalal_HOG();
        }

        public void set_params_dalal_HOG()
        {
            binSize = 8;
            nOrients = 9;
            clipHog = 0.2f;
            softBin = 1;
            useHog = 1;
            full_angle = false;
        }

        public void set_params_falzen_HOG()
        {
            binSize = 8;
            nOrients = 9;
            clipHog = 0.2f;
            softBin = -1;
            useHog = 2;
            full_angle = true;
        }

        public void set_param_binSize(int binSize_)
        {
            binSize = binSize_;
        }

        public void set_params_custom(int binSize_, int nOrients_, float clipHog_, int softBin_, int useHog_, boolean full_angle_)
        {
            binSize = binSize_;
            nOrients = nOrients_;
            clipHog = clipHog_;
            softBin = softBin_;
            useHog = useHog_;
            full_angle = full_angle_;
        }

        public int nchannels_hog()
        {
            return useHog == 0 ? nOrients : (useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
        }

        public int nrows_hog(int nrows_img)
        {
            return nrows_img / binSize;
        }

        public int ncols_hog(int ncols_img)
        {
            return ncols_img / binSize;
        }

        public Results_HogDollar_GradsRaw extract_grads_raw(Matkc img)
        {
            return gradMO_raw(img, full_angle);
        }

        public Matkc extract(Matkc img)
        {
            return gradHist_hog_fhog(img, useHog, binSize, nOrients, softBin, full_angle, clipHog);
        }
    }
    private static native Results_HogDollar_GradsRaw gradMO_raw(Matkc img, boolean full_angle);
    private static native Matkc gradHist_hog_fhog(Matkc img, int useHog, int binSize, int nOrients, int softBin, boolean full_angle, float clipHog);

    public static native double[] vl_svm_train(Matkc train_data, Matkc labels, double lambda, boolean SGD, boolean weight_bal_classes);

    public static native Matkc vl_lbp_extract(Matkc img, int cellSize);

    public static native Matkc vl_hog_extract(Matkc img, int cellSize, boolean dalal_hog, int numOrientations, boolean bilinearOrientations);

    public static Matkc vl_hog_extract(Matkc img, int cellSize, boolean dalal_hog, int numOrientations)
    {
        return vl_hog_extract(img, cellSize, dalal_hog, numOrientations, false);
    }

    public static Matkc vl_hog_extract(Matkc img, int cellSize, boolean dalal_hog)
    {
        return vl_hog_extract(img, cellSize, dalal_hog, 9, false);
    }

    public static Matkc vl_hog_extract(Matkc img, int cellSize)
    {
        return vl_hog_extract(img, cellSize, false, 9, false);
    }

    public static class Results_vl_kmeans
    {
        public Matkc centroids;
        public Matkc labels_clusters;

        public Results_vl_kmeans(Matkc centroids, Matkc labels_clusters)
        {
            this.centroids = centroids; this.labels_clusters = labels_clusters;
        }
    }

    public static enum vl_kmeans_alg {
        VlKMeansLloyd,       /**< Lloyd algorithm */
        VlKMeansElkan,       /**< Elkan algorithm */
        VlKMeansANN          /**< Approximate nearest neighbors */
    };

    public static enum kmeans_distType {
        VlDistanceL1,        /**< l1 distance (squared intersection metric) */
        VlDistanceL2,        /**< squared l2 distance */
        VlDistanceChi2,      /**< squared Chi2 distance */
        VlDistanceHellinger, /**< squared Hellinger's distance */
        VlDistanceJS,        /**< squared Jensen-Shannon distance */
        VlDistanceMahalanobis,     /**< squared mahalanobis distance */
        VlKernelL1,          /**< intersection kernel */
        VlKernelL2,          /**< l2 kernel */
        VlKernelChi2,        /**< Chi2 kernel */
        VlKernelHellinger,   /**< Hellinger's kernel */
        VlKernelJS           /**< Jensen-Shannon kernel */
    };

    public static enum kmeans_iniMethod {
        VlKMeansRandomSelection,  /**< Randomized selection */
        VlKMeansPlusPlus          /**< Plus plus raondomized selection */
    };

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters, int num_repeats, kmeans_iniMethod init_method, int verbosity, int num_comparisons, int ntrees)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, num_repeats, init_method.ordinal(), verbosity, num_comparisons, ntrees);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters, int num_repeats, kmeans_iniMethod init_method, int verbosity, int num_comparisons)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, num_repeats, init_method.ordinal(), verbosity, num_comparisons, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters, int num_repeats, kmeans_iniMethod init_method, int verbosity)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, num_repeats, init_method.ordinal(), verbosity, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters, int num_repeats, kmeans_iniMethod init_method)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, num_repeats, init_method.ordinal(), 0, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters, int num_repeats)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, num_repeats, kmeans_iniMethod.VlKMeansPlusPlus.ordinal(), 0, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type, int maxIters)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), maxIters, 1, kmeans_iniMethod.VlKMeansPlusPlus.ordinal(), 0, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg, kmeans_distType dist_type)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), dist_type.ordinal(), 100, 1, kmeans_iniMethod.VlKMeansPlusPlus.ordinal(), 0, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, vl_kmeans_alg alg)
    {
        return vl_kmeans(train_data, nclusters, alg.ordinal(), kmeans_distType.VlDistanceL2.ordinal(), 100, 1, kmeans_iniMethod.VlKMeansPlusPlus.ordinal(), 0, 100, 3);
    }

    public static Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters)
    {
        return vl_kmeans(train_data, nclusters, vl_kmeans_alg.VlKMeansElkan.ordinal(), kmeans_distType.VlDistanceL2.ordinal(), 100, 1, kmeans_iniMethod.VlKMeansPlusPlus.ordinal(), 0, 100, 3);
    }

    private static native Results_vl_kmeans vl_kmeans(Matkc train_data, int nclusters, int alg, int dist_type, int maxIters, int num_repeats, int init_method, int verbosity, int num_comparisons, int ntrees);

    public static class Results_vl_sift
    {
        public Matkc descriptors;
        public Matkc keypoints;

        public Results_vl_sift(Matkc descriptors, Matkc keypoints)
        {
            this.descriptors = descriptors; this.keypoints = keypoints;
        }
    }

    public static class vl_sift
    {
        private int octaves;
        private int levels;
        private int firstOctave;
        private double peakThresh;
        private double edgeThresh;
        private double normThresh;
        private double magnif;
        private double windowSize;
        private Matkc frames;
        private boolean computeOrientations;
        private int verbose;

        public vl_sift()
        {
            octaves = -1;
            levels = 3;
            firstOctave = 0;
            peakThresh = -1;
            edgeThresh = -1;
            normThresh = -1;
            magnif = -1;
            windowSize = -1;
            frames = new Matkc();
            computeOrientations = false;
            verbose = 0;
        }

        public void set_octaves(int octaves_)
        {
            octaves= octaves_;
        }

        public void set_levels(int levels_)
        {
            levels = levels_;
        }

        public void set_firstOctave(int firstOctave_)
        {
            firstOctave = firstOctave_;
        }

        public void set_peakThresh(double peakThresh_)
        {
            peakThresh = peakThresh_;
        }

        public void set_edgeThresh(double edgeThresh_)
        {
            edgeThresh = edgeThresh_;
        }

        public void set_normThresh(double normThresh_)
        {
            normThresh = normThresh_;
        }

        public void set_magnif(double magnif_)
        {
            magnif = magnif_;
        }

        public void set_windowSize(double windowSize_)
        {
            windowSize = windowSize_;
        }

        public void set_frames(Matkc frames_)
        {
            frames = frames_;
        }

        public void set_computeOrientations(boolean computeOrientations_)
        {
            computeOrientations = computeOrientations_;
        }

        public void set_verbose(int verbose_)
        {
            verbose = verbose;
        }

        public Results_vl_sift extract(Matkc img)
        {
            if(img.nchannels() != 1)
                throw new IllegalArgumentException("ERROR: img must have 1 channel.");
            return vl_sift_jni(img, octaves, levels, firstOctave, peakThresh, edgeThresh, normThresh, magnif, windowSize, frames, computeOrientations, verbose);
        }
    }

    private static native Results_vl_sift vl_sift_jni(Matkc img, int octaves, int levels, int firstOctave, double peakThresh, double edgeThresh, double normThresh, double magnif, double windowSize, Matkc frames, boolean computeOrientations, int verbose);

    public static class vl_dsift
    {
        private int[] step;
        private int[] size;
        private double[] bounds;
        private boolean output_norm;
        private boolean fast;
        private int[] geometry;
        private boolean verbose;

        public vl_dsift()
        {
            step = new int[2]; step[0] = 1; step[1] = 1;
            size = new int[2]; size[0] = 3; size[1] = 3;
            bounds = new double[4]; for(int i=0; i<4; i++) bounds[i] = -1; // if all -1, then it is a special notation for "the whole image"
            output_norm = false;
            fast = false;
            geometry = new int[3]; geometry[0] = 4; geometry[1] = 4; geometry[2] = 8;
            verbose = false;
        }

        public void set_step(int[] step_)
        {
            if(step_.length != 2)
                throw new IllegalArgumentException("ERROR: step must be an array of length 2.");
            if (step_[0] < 1 || step_[1] < 1)
                throw new IllegalArgumentException("ERROR: step values are invalid.");
            step = step_;
        }

        public void set_size(int[] size_)
        {
            if(size_.length != 2)
                throw new IllegalArgumentException("ERROR: size must be an array of length 2.");
            if(size_[0] < 1 || size_[1] < 1)
                throw new IllegalArgumentException("ERROR: size values are invalid.");
            size = size_;
        }

        public void set_bounds(double[] bounds_)
        {
            if(bounds.length != 4)
                throw new IllegalArgumentException("ERROR: bounds must be an array of length 4: [XMIN, YMIN, XMAX, YMAX].");
            bounds = bounds_;
        }

        public void set_output_norm(boolean output_norm_)
        {
            output_norm = output_norm_;
        }

        public void set_fast(boolean fast_)
        {
            fast = fast_;
        }

        public void set_geometry(int[] geometry_)
        {
            if(geometry.length != 3)
                throw new IllegalArgumentException("ERROR: bounds must be an array of length 3: [NX NY NO], where NX is the number of bin in the X direction, NY in the Y direction, and NO the nubmer of orientation bins.");
            if (geometry[0] < 1 || geometry[1] < 1 || geometry[2] < 1)
                throw new IllegalArgumentException("ERROR: geometry values are invalid.");
            geometry = geometry_;
        }

        public void set_verbose(boolean verbose_)
        {
            verbose = verbose_;
        }

        public Results_vl_sift extract(Matkc img)
        {
            if(img.nchannels() != 1)
                throw new IllegalArgumentException("ERROR: img must have 1 channel.");
            return vl_dsift_jni(img, step, size, bounds, output_norm, fast, geometry, verbose);
        }
    }

    private static native Results_vl_sift vl_dsift_jni(Matkc img, int[] step, int[] size, double[] bounds, boolean output_norm, boolean fast, int[] geometry, boolean verbose);


}
