#pragma once

#include "./activation_function_layer_module.h"

int cnn_sigmoid_layer_forward(struct cnn_Layer *layer, int index, int max_index);
int cnn_sigmoid_layer_backward(struct cnn_Layer *layer, int index, int max_index);