#pragma once
#include "./../../layer_module.h"

void cnn_fully_connected_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index);
void cnn_fully_connected_layer_forward(struct cnn_Layer *layer, int index, int max_index);
void cnn_fully_connected_layer_backward(struct cnn_Layer *layer, int index, int max_index);
void cnn_fully_connected_layer_initForward(struct cnn_Layer *layer);
void cnn_fully_connected_layer_initBackward(struct cnn_Layer *layer);

void cnn_fully_connected_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b);
void cnn_fully_connected_layer_releaseInnerData(struct cnn_Layer* layer);