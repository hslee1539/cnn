#include "./activation_function_layer_module.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* out = layer->out;
struct Tensor* dx = layer->dx;
struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
*/

int cnn_activation_function_layer_initForward(struct cnn_Layer *layer){
    struct Tensor* x = layer->inLayer[0]->out;
    if(layer->out->shapes[0] != x->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = tensor_create_struct_deep(x);
    }
    return 0;
}

int cnn_activation_function_layer_initBackward(struct cnn_Layer *layer){
    struct Tensor* dout = layer->outLayer[0]->dx;
    if(layer->dx->shapes[0] != dout->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(dout);
    }
    return 0;
}