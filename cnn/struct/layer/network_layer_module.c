#include "./network_layer_module.h"
#include "./function/network_layer_module.h"

char cnn_NETWORK_LAYER_NAME[] = "network";

int cnn_isNetworkLayer(struct cnn_Layer *layer){
    return layer->name == cnn_NETWORK_LAYER_NAME;
}

struct cnn_Layer* cnn_create_network_layer(int size){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_NETWORK_LAYER_NAME, size, cnn_network_layer_forward, cnn_network_layer_backward, cnn_network_layer_initForward, cnn_network_layer_initBackward, cnn_network_layer_releaseInnerData, cnn_network_layer_update);
    cnn_network_layer_createInnerData(newLayer);
    return newLayer;
}
