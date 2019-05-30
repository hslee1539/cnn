#include "./activation_function_layer_module.h"
#include "standard_layer_define.h"
/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* out = layer->out;
struct Tensor* dx = layer->dx;
struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
*/

int cnn_activation_function_layer_initForward(struct cnn_Layer *layer){
    if(layer->out->shapes[0] != CNN_LAYER_X(layer)->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = tensor_create_struct_deep(CNN_LAYER_X(layer));
    }
    return 0;
}

int cnn_activation_function_layer_initBackward(struct cnn_Layer *layer){
    struct Tensor* dout = layer->outLayer[0]->dx;
    if(layer->dx->shapes[0] != CNN_LAYER_DOUT(layer)->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(CNN_LAYER_DOUT(layer));
    }
    return 0;
}