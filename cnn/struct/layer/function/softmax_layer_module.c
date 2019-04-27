#include <malloc.h>

#include "./softmax_layer_module.h"
#include "./computing/softmax_layer_module.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
struct Tensor* table = layer->outLayer[1]->dx;
*/

int cnn_softmax_crossentropy_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* x = layer->inLayer[0]->out;
    cnn_comput_softmax_crossentropy_layer_forward(x, layer->out, index, max_index);
    return 0;
}
int cnn_softmax_crossentropy_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* dout = layer->outLayer[0]->dx;
    struct Tensor* table = layer->outLayer[1]->dx;
    cnn_comput_softmax_crossentropy_layer_backward(dout, layer->out, table, layer->dx, index, max_index);
    return 0;
}