#pragma once

#include "./../layer_module.h"
//TODO 내부 변수 생성 안함. 생성하는 코드 추가 할 것
struct cnn_Layer *cnn_create_softmax_crossentropy_layer(struct cnn_Layer *in, struct cnn_Layer *dataset);