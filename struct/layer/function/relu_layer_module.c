//#include <malloc.h>

#include "relu_layer_module.h"
#include "./../../../computing/relu_layer_module.h"
#include "./standard_layer_define.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
*/

void cnn_relu_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_relu_layer_forward(CNN_LAYER_X(layer), layer->out, index, max_index);
}

void cnn_relu_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_relu_layer_backward(CNN_LAYER_DOUT(layer), layer->out, layer->dx, index, max_index);
}
