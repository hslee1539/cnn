#pragma once
#include "./../../layer_module.h"

int cnn_batchnorm_layer_forward(struct cnn_Layer *layer, int index, int max_index);
int cnn_batchnorm_layer_backward(struct cnn_Layer *layer, int index, int max_index);
int cnn_batchnorm_layer_initForward(struct cnn_Layer *layer);
int cnn_batchnorm_layer_initBackward(struct cnn_Layer *layer);

int cnn_batchnorm_layer_createInnerData(struct cnn_Layer *layer);
int cnn_batchnorm_layer_releaseInnerData(struct cnn_Layer *layer);