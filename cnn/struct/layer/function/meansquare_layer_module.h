#pragma once
#include "./activation_function_layer_module.h"

// initForward는 항상 out이 스칼라 값이므로 필요가 없음(loss값 저장)
// initBackward는 activation_function의 initBackward 사용
//

int cnn_meansquare_layer_forward(struct cnn_Layer *layer, int index, int max_index);
int cnn_meansquare_layer_backward(struct cnn_Layer *layer, int index, int max_index);
int cnn_meansquare_layer_initBackward(struct cnn_Layer *layer);