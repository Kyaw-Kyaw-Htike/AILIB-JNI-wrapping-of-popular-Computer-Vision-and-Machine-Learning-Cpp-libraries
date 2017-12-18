#pragma once

#include "sift.h"

void transpose_descriptor(float* dst, float* src);
int korder(void const* a, void const* b);
vl_bool check_sorted(double const * keys, vl_size nkeys);