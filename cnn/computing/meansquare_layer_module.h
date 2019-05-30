#pragma once
#include "../../tensor/main_module.h"

void cnn_comput_meansquare_layer_forward(struct Tensor* x, struct Tensor* table, struct Tensor* out, int index, int max_index);
void cnn_comput_meansquare_layer_backward(struct Tensor* x, struct Tensor* table, struct Tensor* dx, int index, int max_index);

