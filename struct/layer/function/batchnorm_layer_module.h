#pragma once
#define CNN_BATCHNORM_LAYER_IDSPERSION(layer) layer->extra->tensors[0]
#include "./../../layer_module.h"

void cnn_batchnorm_layer_forward(struct cnn_Layer *layer, int index, int max_index);
void cnn_batchnorm_layer_backward(struct cnn_Layer *layer, int index, int max_index);
void cnn_batchnorm_layer_initForward(struct cnn_Layer *layer);
void cnn_batchnorm_layer_initBackward(struct cnn_Layer *layer);

void cnn_batchnorm_layer_createInnerData(struct cnn_Layer *layer);
void cnn_batchnorm_layer_releaseInnerData(struct cnn_Layer *layer);