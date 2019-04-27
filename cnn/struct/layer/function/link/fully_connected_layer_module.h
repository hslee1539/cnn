#pragma once

#include "./../../../layer_module.h"

void cnn_fully_connected_relu_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b);
void cnn_fully_connected_relu_layer_releaseInnerData(struct cnn_Layer* layer);

void cnn_fully_connected_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b);
void cnn_fully_connected_layer_releaseInnerData(struct cnn_Layer* layer);

struct Tensor* cnn_fully_connected_layer_get_w(struct cnn_Layer *layer);
struct Tensor* cnn_fully_connected_layer_get_b(struct cnn_Layer *layer);
struct Tensor* cnn_fully_connected_layer_get_dw(struct cnn_Layer *layer);
struct Tensor* cnn_fully_connected_layer_get_db(struct cnn_Layer *layer);
struct Tensor* cnn_fully_connected_layer_get_activation_dx(struct cnn_Layer *layer);
// release를 한 후, 얕은 복사(객체 포인터 복사)를 합니다.
void cnn_fully_connected_layer_set_w(struct cnn_Layer *layer, struct Tensor *source);
// release를 한 후, 얕은 복사(객체 포인터 복사)를 합니다.
void cnn_fully_connected_layer_set_b(struct cnn_Layer *layer, struct Tensor *source);
// release를 한 후, 얕은 복사(객체 포인터 복사)를 합니다.
void cnn_fully_connected_layer_set_dw(struct cnn_Layer *layer, struct Tensor *source);
// release를 한 후, 얕은 복사(객체 포인터 복사)를 합니다.
void cnn_fully_connected_layer_set_db(struct cnn_Layer *layer, struct Tensor *source);
// release를 한 후, 얕은 복사(객체 포인터 복사)를 합니다.
void cnn_fully_connected_layer_set_activation_dx(struct cnn_Layer *layer, struct Tensor *source);