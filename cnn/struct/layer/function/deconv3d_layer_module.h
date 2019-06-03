#pragma once
#include "./../../layer_module.h"

int cnn_deconv3d_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index);
int cnn_deconv3d_layer_forward(struct cnn_Layer *layer, int index, int max_index);
int cnn_deconv3d_layer_backward(struct cnn_Layer *layer, int index, int max_index);
int cnn_deconv3d_layer_initForward(struct cnn_Layer *layer);
int cnn_deconv3d_layer_initBackward(struct cnn_Layer *layer);

int cnn_deconv3d_layer_createInnerData(struct cnn_Layer *layer, struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding);

int cnn_deconv3d_layer_releaseInnerData(struct cnn_Layer *layer);