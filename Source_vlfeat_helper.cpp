#include "sift.h"

void transpose_descriptor(float* dst, float* src)
{
	int const BO = 8;  /* number of orientation bins */
	int const BP = 4;  /* number of spatial bins     */
	int i, j, t;

	for (j = 0; j < BP; ++j) {
		int jp = BP - 1 - j;
		for (i = 0; i < BP; ++i) {
			int o = BO * i + BP*BO * j;
			int op = BO * i + BP*BO * jp;
			dst[op] = src[o];
			for (t = 1; t < BO; ++t)
				dst[BO - t + op] = src[t + o];
		}
	}
}

/** ------------------------------------------------------------------
** @internal
** @brief Ordering of tuples by increasing scale
**
** @param a tuple.
** @param b tuple.
**
** @return @c a[2] < b[2]
**/

int korder(void const* a, void const* b) {
	double x = ((double*)a)[2] - ((double*)b)[2];
	if (x < 0) return -1;
	if (x > 0) return +1;
	return 0;
}

/** ------------------------------------------------------------------
** @internal
** @brief Check for sorted keypoints
**
** @param keys keypoint list to check
** @param nkeys size of the list.
**
** @return 1 if the keypoints are storted.
**/

vl_bool check_sorted(double const * keys, vl_size nkeys)
{
	vl_uindex k;
	for (k = 0; k + 1 < nkeys; ++k) {
		if (korder(keys, keys + 4) > 0) {
			return VL_FALSE;
		}
		keys += 4;
	}
	return VL_TRUE;
}