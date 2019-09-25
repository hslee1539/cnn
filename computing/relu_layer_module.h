#pragma once
#include "../tensor/import_module.h"

void cnn_comput_relu_layer_forward(struct Tensor* x, struct Tensor* out, int index, int max_index);
void cnn_comput_relu_layer_backward(struct Tensor* dout, struct Tensor* out, struct Tensor* dx, int index, int max_index);
