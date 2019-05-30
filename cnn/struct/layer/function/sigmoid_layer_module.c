#include <malloc.h>

#include "./sigmoid_layer_module.h"
#include "./../../../computing/sigmoid_layer_module.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
*/

int cnn_sigmoid_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* x = layer->inLayer[0]->out;
    cnn_comput_sigmoid_layer_forward(x, layer->out, index, max_index);
    return 0;
}
int cnn_sigmoid_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* dout = layer->outLayer[0]->dx;
    cnn_comput_sigmoid_layer_backward(dout, layer->out, layer->dx, index, max_index);
    return 0;
}