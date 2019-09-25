#pragma once
#define CNN_NETWORK_LAYER_INDEX(layer) layer->extra->ints[0]
#include "./../../layer_module.h"

void cnn_network_layer_createInnerData(struct cnn_Layer* layer);
void cnn_network_layer_releaseInnerData(struct cnn_Layer* layer);

void cnn_network_layer_forward(struct cnn_Layer *layer, int index, int max_index);
void cnn_network_layer_backward(struct cnn_Layer *layer, int index, int max_index);
void cnn_network_layer_initForward(struct cnn_Layer *layer);
void cnn_network_layer_initBackward(struct cnn_Layer *layer);
void cnn_network_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index);
