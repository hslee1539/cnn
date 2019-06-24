
// 뼈대 해더
#include "./fully_connected_layer_module.h"
// private 해더
#include "./function/fully_connected_layer_module.h"

char cnn_FULLY_CONNECTED_LAYER_NAME[] = "fully connected layer";

struct cnn_Layer *cnn_create_fully_connected_layer(struct Tensor *w, struct Tensor *b){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_FULLY_CONNECTED_LAYER_NAME, 0, cnn_fully_connected_layer_forward, cnn_fully_connected_layer_backward, cnn_fully_connected_layer_initForward, cnn_fully_connected_layer_initBackward, cnn_fully_connected_layer_releaseInnerData, cnn_fully_connected_layer_update, _cnn_layer_baseInitUpdate);
    cnn_fully_connected_layer_createInnerData(newLayer, w, b);
    return newLayer;
}