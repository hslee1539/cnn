#pragma once

#include "./../layer_module.h"

int cnn_isNetworkLayer(struct cnn_Layer *layer);
struct cnn_Layer *cnn_create_network_layer(int size);