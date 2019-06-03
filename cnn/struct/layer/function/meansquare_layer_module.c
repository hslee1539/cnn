#include "./meansquare_layer_module.h"
#include "./../../../computing/meansquare_layer_module.h"
#include "./standard_last_layer_define.h"

int cnn_meansquare_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_meansquare_layer_forward(CNN_LAYER_X(layer), CNN_LAYER_TABLE(layer), CNN_LAYER_OUT(layer), index, max_index);
    return 0;
}

int cnn_meansquare_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_meansquare_layer_backward(CNN_LAYER_X(layer), CNN_LAYER_TABLE(layer), CNN_LAYER_DX(layer), index, max_index);
    return 0;
}

int cnn_meansquare_layer_initBackward(struct cnn_Layer *layer){
    if(layer->dx->shapes[0] != CNN_LAYER_X(layer)->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(CNN_LAYER_X(layer));
    }
    return 0;
}