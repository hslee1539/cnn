#pragma once

#include "./../../layer_module.h"

int cnn_network_create(struct cnn_Layer* layer);
int cnn_network_release(struct cnn_Layer* layer);

int cnn_network_forward(struct cnn_Layer* layer, int index, int max_index);
int cnn_network_backward(struct cnn_Layer* layer, int index, int max_index);
int cnn_network_initForward(struct cnn_Layer* layer);
int cnn_network_initBackward(struct cnn_Layer* layer);
int cnn_network_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index);
