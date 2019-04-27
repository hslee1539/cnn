#pragma once
#include "../../tensor/main_module.h"

struct Tensor* cnn_create_fully_connected_out(struct Tensor* x, struct Tensor* w);

void cnn_comput_fully_connected_relu_layer_forward(struct Tensor* x, struct Tensor* w, struct Tensor* b, struct Tensor* out, int index, int max_index);
//dx가 갱신, activation_dx은 임시 개산 저장 공간이고 update할때 활용
void cnn_comput_fully_connected_relu_layer_backward(struct Tensor* dout, struct Tensor* w, struct Tensor* out, struct Tensor* activation_dx, struct Tensor* dx, int index, int max_index);
void cnn_comput_fully_connected_relu_layer_dw(struct Tensor* x, struct Tensor* activation_dx, struct Tensor* dw, int index, int max_index);
void cnn_comput_fully_connected_relu_layer_db(struct Tensor* activation_dx, struct Tensor* db, int index, int max_index);

void cnn_comput_fully_connected_layer_forward(struct Tensor* x, struct Tensor* w, struct Tensor* b, struct Tensor* out, int index, int max_index);
void cnn_comput_fully_connected_layer_backward(struct Tensor* dout, struct Tensor* w, struct Tensor* dx, int index, int max_index);
void cnn_comput_fully_connected_layer_dw(struct Tensor* dout, struct Tensor* x, struct Tensor* dw, int index, int max_index);
void cnn_comput_fully_connected_layer_db(struct Tensor* dout, struct Tensor* db, int index, int max_index);