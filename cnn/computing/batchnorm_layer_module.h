#pragma once
#include "../../tensor/main_module.h"

void cnn_comput_batchnorm_layer_forward(struct Tensor *x, struct Tensor *dispersion, struct Tensor *out, int index, int max_index);
void cnn_comput_batchnorm_layer_backward(struct Tensor *dout, struct Tensor *out, struct Tensor *dispersion, struct Tensor *dx, int index, int max_index);