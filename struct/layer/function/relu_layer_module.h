#pragma once

#include "./activation_function_layer_module.h"

void cnn_relu_layer_forward(struct cnn_Layer *layer, int index, int max_index);
void cnn_relu_layer_backward(struct cnn_Layer *layer, int index, int max_index);