#pragma once

// 활성함수들의 공통되는 함수들의 묶음

#include "./../../layer_module.h"

void cnn_activation_function_layer_initForward(struct cnn_Layer *layer);
void cnn_activation_function_layer_initBackward(struct cnn_Layer *layer);