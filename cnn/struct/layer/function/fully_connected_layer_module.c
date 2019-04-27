#include "./fully_connected_layer_module.h"
#include "./computing/fully_connected_layer_module.h"

/*
코딩 실수 줄이기 위해 복사해서 사용하자!

struct Tensor* out = layer->out;
struct Tensor* dx = layer->dx;
struct Tensor* x = layer->inLayer[0]->out;
struct Tensor* dout = layer->outLayer[0]->dx;
struct Tensor* w = layer->updateList->sets[0]->value;
struct Tensor* dw = layer->updateList->sets[0]->delta;
struct Tensor* b = layer->updateList->sets[1]->value;
struct Tensor* db = layer->updateList->sets[1]->delta;
*/

int cnn_fully_connected_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    struct Tensor* x = layer->inLayer[0]->out;
    struct Tensor* dw = layer->updateList->sets[0]->delta;
    struct Tensor* dout = layer->outLayer[0]->dx;
    struct Tensor* db = layer->updateList->sets[1]->delta;

    cnn_comput_fully_connected_layer_dw(dout, x, dw, index, max_index);
    cnn_comput_fully_connected_layer_db(dout, db, index, max_index);
    cnn_optimizer_update(optimizer, layer->updateList, index, max_index);
    return 0;
}
int cnn_fully_connected_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* x = layer->inLayer[0]->out;
    struct Tensor* w = layer->updateList->sets[0]->value;
    struct Tensor* b = layer->updateList->sets[1]->value;

    cnn_comput_fully_connected_layer_forward(x, w, b, layer->out, index, max_index);
    return 0;
}
int cnn_fully_connected_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    struct Tensor* dout = layer->outLayer[0]->dx;
    struct Tensor* w = layer->updateList->sets[0]->value;
    struct Tensor* dx = layer->dx;

    cnn_comput_fully_connected_layer_backward(dout, w, dx, index, max_index);
    return 0;
}

int cnn_fully_connected_layer_initForward(struct cnn_Layer *layer){
    struct Tensor* x = layer->inLayer[0]->out;
    struct Tensor* w = layer->updateList->sets[0]->value;

    if(layer->out->shapes[0] != x->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = cnn_create_fully_connected_out(x, w);
    }
    return 0;
}
int cnn_fully_connected_layer_initBackward(struct cnn_Layer *layer){
    struct Tensor* x = layer->inLayer[0]->out;
    struct Tensor* dout = layer->outLayer[0]->dx;

    if(layer->dx->shapes[0] != dout->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(x);
    }
    return 0;
}

int cnn_fully_connected_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b){
    layer->updateList = cnn_create_updatelist(2);
    layer->updateList->sets[0] = cnn_create_updateset(tensor_create_struct_deep(w), w);
    layer->updateList->sets[1] = cnn_create_updateset(tensor_create_struct_deep(b), b);
    return 0;
}
int cnn_fully_connected_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_updatelist_deep(layer->updateList);
    return 0;
}