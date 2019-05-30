#pragma once

#define CNN_LAYER_OUT(layer) layer->out
#define CNN_LAYER_X(layer) layer->inLayer[0]->out
#define CNN_LAYER_DOUT(layer) layer->outLayer[0]->dx
#define CNN_LAYER_DX(layer) layer->dx
