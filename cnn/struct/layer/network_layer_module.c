#include "./network_layer_module.h"
#include "./function/network_layer_module.h"

char cnn_NETWORK_LAYER_NAME[] = "network";

int cnn_isNetworkLayer(struct cnn_Layer *layer){
    return layer->name == cnn_NETWORK_LAYER_NAME;
}

struct cnn_Layer* cnn_create_network_layer(int size){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_NETWORK_LAYER_NAME, size, cnn_network_layer_forward, cnn_network_layer_backward, cnn_network_layer_initForward, cnn_network_layer_initBackward, cnn_network_layer_releaseInnerData, cnn_network_layer_update, _cnn_layer_baseInitUpdate);
    cnn_network_layer_createInnerData(newLayer);
    return newLayer;
}

int cnn_network_next(struct cnn_Layer *layer){
    if(cnn_isNetworkLayer(layer)){
        int layer_index = CNN_NETWORK_LAYER_INDEX(layer);
        int retval = cnn_network_next(layer->childLayer[layer_index]);
        if(retval == 0){
            CNN_NETWORK_LAYER_INDEX(layer)++;
            if(CNN_NETWORK_LAYER_INDEX(layer) == layer->childLayer_size)
                CNN_NETWORK_LAYER_INDEX(layer) = 0;
        }
        return retval + layer->childLayer_size - layer_index - 1;
    }
    return 0;
}