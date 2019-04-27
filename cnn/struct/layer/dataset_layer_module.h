#pragma once

#include "./../layer_module.h"

///x와 table을 참조하여 데이터 셋 레이어를 만듭니다. out이 x이고, dx가 table로 사용됩니다.
struct cnn_Layer* cnn_create_dataset_layer(struct Tensor* x, struct Tensor* table);