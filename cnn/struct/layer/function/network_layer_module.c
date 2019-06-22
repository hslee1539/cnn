#include "./network_layer_module.h"

void cnn_network_layer_createInnerData(struct cnn_Layer* layer){
    layer->extra = cnn_create_extraData(0,0,1);
    CNN_NETWORK_LAYER_INDEX(layer) = 0;
}

void cnn_network_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_extradata(layer->extra);
}

void cnn_network_layer_forward(struct cnn_Layer* layer, int index, int max_index){
    cnn_layer_forward(layer->childLayer[CNN_NETWORK_LAYER_INDEX(layer)], index, max_index);
}

void cnn_network_layer_backward(struct cnn_Layer* layer, int index, int max_index){
    cnn_layer_backward(layer->childLayer[layer->childLayer_size - CNN_NETWORK_LAYER_INDEX(layer) - 1], index, max_index);
}

void cnn_network_layer_initForward(struct cnn_Layer *layer){
    for(int i = 0; i < layer->childLayer_size; i ++){
        cnn_layer_initForward(layer->childLayer[i]);
    }
    tensor_referTo(layer->out, cnn_layer_getRightTerminal(layer)->out);
}

void cnn_network_layer_initBackward(struct cnn_Layer *layer){
    for(int i = layer->childLayer_size - 1; i > -1; i--)
        cnn_layer_initBackward(layer->childLayer[i]);
    tensor_referTo(layer->dx, cnn_layer_getLeftTerminal(layer)->dx);
}

void cnn_network_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    cnn_layer_update(layer->childLayer[CNN_NETWORK_LAYER_INDEX(layer)], optimizer, index, max_index);
}