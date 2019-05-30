#include <malloc.h>

#include "./softmax_layer_module.h"
#include "./../../../computing/softmax_layer_module.h"
#include "./standard_last_layer_define.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
struct Tensor* table = layer->outLayer[1]->dx;
*/

int cnn_softmax_crossentropy_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_softmax_crossentropy_layer_forward(CNN_LAYER_X(layer), layer->out, index, max_index);
    return 0;
}

int cnn_softmax_crossentropy_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_softmax_crossentropy_layer_backward(layer->out, CNN_LAYER_TABLE(layer), layer->dx, index, max_index);
    return 0;
}