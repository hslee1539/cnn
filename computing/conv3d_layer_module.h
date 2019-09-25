#pragma once

#include "../../tensor/main_module.h"

struct Tensor *cnn_create_conv3d_layer_out(struct Tensor *x, struct Tensor *filter, int stride, int pad);

void cnn_comput_conv3d_layer_forward(struct Tensor *x, struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding, struct Tensor *out, int index, int max_index);
void cnn_comput_conv3d_layer_backward(struct Tensor *dout, struct Tensor *filter, int stride, int pad, struct Tensor *dx, int index, int max_index);
void cnn_comput_conv3d_layer_dfilter(struct Tensor *dout, struct Tensor *x, int stride, int pad, int padding, struct Tensor *dfilter, int index, int max_index);
void cnn_comput_conv3d_layer_dbias(struct Tensor *dout, struct Tensor *dbias, int index, int max_index);