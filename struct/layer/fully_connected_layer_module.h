#pragma once

#include "./../layer_module.h"

struct cnn_Layer *cnn_create_fully_connected_layer(struct Tensor *w, struct Tensor *b);