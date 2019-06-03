#pragma once

#define CNN_LAYER_OUT(layer) layer->out
#define CNN_LAYER_X(layer) layer->inLayer->out
#define CNN_LAYER_DOUT(layer) layer->outLayer->dx
#define CNN_LAYER_DX(layer) layer->dx
