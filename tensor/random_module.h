#pragma once
#include "./struct/tensor_module.h"

struct Tensor* tensor_create_gauss_deep(long long *shapes, int dim, int seed);
struct Tensor* tensor_create_random_deep(long long *shapes, int dim, int seed, float min, float range);
void tensor_gauss(struct Tensor* tensor, int seed);
void tensor_random(struct Tensor* tensor, int seed, float min, float range);
void tensor_shuffle(struct Tensor* tensor, int seed);
