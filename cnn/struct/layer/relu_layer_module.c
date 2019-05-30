#include "./relu_layer_module.h"

#include "./function/relu_layer_module.h"

char cnn_RELU_LAYER_NAME[] = "relu";

struct cnn_Layer* cnn_create_relu_layer(struct cnn_Layer* in, struct cnn_Layer* out){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_RELU_LAYER_NAME, 1, 1, cnn_relu_layer_forward, cnn_relu_layer_backward, cnn_activation_function_layer_initForward, cnn_activation_function_layer_initBackward, 0, 0);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = out;
    return newLayer;
}