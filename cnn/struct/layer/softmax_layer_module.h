#pragma once

#include "./../layer_module.h"

struct cnn_Layer* cnn_create_softmax_crossentropy_layer(struct cnn_Layer* in, struct cnn_Layer* out);