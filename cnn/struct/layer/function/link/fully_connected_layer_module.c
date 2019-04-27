#include "./fully_connected_layer_module.h"

void cnn_fully_connected_relu_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b){
    layer->extra = cnn_create_extraData(1,0,0); 
    layer->extra->tensors[0] = tensor_create();
    layer->value = cnn_create_updatelist(2);
    layer->value->sets[0] = cnn_create_updateset(tensor_create_struct_deep(w), w);
    layer->value->sets[1] = cnn_create_updateset(tensor_create_struct_deep(b), b);

}
void cnn_fully_connected_relu_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_extradata_deep(layer->extra);
    cnn_release_updatelist_deep(layer->value);
}
void cnn_fully_connected_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b){
    layer->value = cnn_create_updatelist(2);
    layer->value->sets[0] = cnn_create_updateset(tensor_create_struct_deep(w), w);
    layer->value->sets[1] = cnn_create_updateset(tensor_create_struct_deep(b), b);
}
void cnn_fully_connected_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_updatelist_deep(layer->value);
}

struct Tensor* cnn_fully_connected_layer_get_w(struct cnn_Layer *layer){
    return layer->value->sets[0]->value;
}
struct Tensor* cnn_fully_connected_layer_get_b(struct cnn_Layer *layer){
    return layer->value->sets[1]->value;
}
struct Tensor* cnn_fully_connected_layer_get_dw(struct cnn_Layer *layer){
    return layer->value->sets[0]->delta;
}
struct Tensor* cnn_fully_connected_layer_get_db(struct cnn_Layer *layer){
    return layer->value->sets[1]->delta;
}
struct Tensor* cnn_fully_connected_layer_get_activation_dx(struct cnn_Layer *layer){
    return layer->extra->tensors[0];
}
void cnn_fully_connected_layer_set_w(struct cnn_Layer *layer, struct Tensor *source){
    tensor_release_deep(layer->value->sets[0]->value);
    layer->value->sets[0]->value = source;
}
void cnn_fully_connected_layer_set_b(struct cnn_Layer *layer, struct Tensor *source){
    tensor_release_deep(layer->value->sets[1]->value);
    layer->value->sets[1]->value = source;
}
void cnn_fully_connected_layer_set_dw(struct cnn_Layer *layer, struct Tensor *source){
    tensor_release_deep(layer->value->sets[0]->delta);
    layer->value->sets[0]->delta = source;
}
void cnn_fully_connected_layer_set_db(struct cnn_Layer *layer, struct Tensor *source){
    tensor_release_deep(layer->value->sets[1]->delta);
    layer->value->sets[1]->delta = source;
}
void cnn_fully_connected_layer_set_activation_dx(struct cnn_Layer *layer, struct Tensor *source){
    tensor_release_deep(layer->extra->tensors[0]);
    layer->extra->tensors[0] = source;
}