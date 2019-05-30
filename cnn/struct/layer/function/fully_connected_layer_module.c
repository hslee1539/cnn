#include "./fully_connected_layer_module.h"
#include "./../../../computing/fully_connected_layer_module.h"
#include "./standard_updatable_layer_define.h"

int cnn_fully_connected_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    cnn_comput_fully_connected_layer_dw(CNN_LAYER_DOUT(layer), CNN_LAYER_X(layer), CNN_LAYER_W(layer)->delta, index, max_index);
    cnn_comput_fully_connected_layer_db(CNN_LAYER_DOUT(layer), CNN_LAYER_B(layer)->delta, index, max_index);
    cnn_optimizer_update(optimizer, layer->updateList, index, max_index);
    return 0;
}
int cnn_fully_connected_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_fully_connected_layer_forward(CNN_LAYER_X(layer), CNN_LAYER_W(layer)->value, CNN_LAYER_B(layer)->value, layer->out, index, max_index);
    return 0;
}
int cnn_fully_connected_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_fully_connected_layer_backward(CNN_LAYER_DOUT(layer), CNN_LAYER_W(layer)->value, CNN_LAYER_DX(layer), index, max_index);
    return 0;
}

int cnn_fully_connected_layer_initForward(struct cnn_Layer *layer){
    if(layer->out->shapes[0] != CNN_LAYER_X(layer)->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = cnn_create_fully_connected_out(CNN_LAYER_X(layer), CNN_LAYER_W(layer)->value);
    }
    return 0;
}
int cnn_fully_connected_layer_initBackward(struct cnn_Layer *layer){
    if(layer->dx->shapes[0] != CNN_LAYER_DOUT(layer)->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(CNN_LAYER_X(layer));
    }
    return 0;
}

int cnn_fully_connected_layer_createInnerData(struct cnn_Layer * layer, struct Tensor* w, struct Tensor* b){
    layer->updateList = cnn_create_updatelist(2);
    CNN_LAYER_W(layer) = cnn_create_updateset(tensor_create_struct_deep(w), w);
    CNN_LAYER_B(layer) = cnn_create_updateset(tensor_create_struct_deep(b), b);
    return 0;
}
int cnn_fully_connected_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_updatelist_deep(layer->updateList);
    return 0;
}