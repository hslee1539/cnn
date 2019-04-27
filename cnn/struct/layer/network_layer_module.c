#include "./network_layer_module.h"
#include "./function/network_layer_module.h"

#define cnn_NETWORK_LAYER_NAME "network"

struct cnn_Layer* cnn_create_network_layer(int size){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_NETWORK_LAYER_NAME, size, 1, cnn_network_forward, cnn_network_backward, cnn_network_initForward, cnn_network_initBackward, cnn_network_release, cnn_network_update);
    cnn_network_create(newLayer);
    return newLayer;
}