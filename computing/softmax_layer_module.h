#pragma once
#include "../tensor/import_module.h"

// TODO gan을 고려해서 다른 네트워크랑 연결되게 짜기.

void cnn_comput_softmax_crossentropy_layer_forward(struct Tensor *x, struct Tensor *out, int index, int max_index);
void cnn_comput_softmax_crossentropy_layer_backward(struct Tensor *out, struct Tensor *table, struct Tensor *dx, int index, int max_index);