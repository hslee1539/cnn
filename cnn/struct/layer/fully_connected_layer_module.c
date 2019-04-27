
// 뼈대 해더
#include "./fully_connected_layer_module.h"
// private 해더
#include "./function/fully_connected_layer_module.h"

#define cnn_FULLY_CONNECTED_LAYER_NAME "fully connected layer"

struct cnn_Layer* cnn_create_fully_connected_layer(struct cnn_Layer* in, struct cnn_Layer* out, struct Tensor* w, struct Tensor* b){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_FULLY_CONNECTED_LAYER_NAME, 0, 0, cnn_fully_connected_layer_forward, cnn_fully_connected_layer_backward, cnn_fully_connected_layer_initForward, cnn_fully_connected_layer_initBackward, cnn_fully_connected_layer_releaseInnerData, cnn_fully_connected_layer_update);
    cnn_fully_connected_layer_createInnerData(newLayer, w, b);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = out;
    return newLayer;
}