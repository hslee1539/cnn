#pragma once
#include "./standard_layer_define.h"

#define CNN_LAYER_W(layer) layer->updateList->sets[0]
#define CNN_LAYER_FILTER(layer) layer->updateList->sets[0]
#define CNN_LAYER_B(layer) layer->updateList->sets[1]
