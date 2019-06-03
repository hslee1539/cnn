#pragma once

#include "./../layer_module.h"

struct cnn_Layer *cnn_create_deconv3d_layer(struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding);
