#pragma once

#include "./../../../layer_module.h"

// TODO loss값을 추가하자!

void cnn_softmax_crossentropy_layer_createInnorData(struct cnn_Layer* layer);
void cnn_softmax_crossentropy_layer_releaseInnorData(struct cnn_Layer* layer);

struct Tensor* cnn_softmax_crossentropy_layer_get_t(struct cnn_Layer* layer);
/// 얕은 복사(객체 포인터 복사)만 합니다. (fully_connected의 set과 다름 주의)
void cnn_softmax_crossentropy_layer_set_t(struct cnn_Layer* layer, struct Tensor* t);

float cnn_softmax_creossentropy_layer_get_loss(struct cnn_Layer* layer);
void cnn_softmax_crossentropy_layer_set_loss(struct cnn_Layer* layer, float  loss);
