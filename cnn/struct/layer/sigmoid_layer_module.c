#include "./sigmoid_layer_module.h"

#include "./function/sigmoid_layer_module.h"

char cnn_SIGMOID_LAYER_NAME[] = "sigmoid";

struct cnn_Layer* cnn_create_sigmoid_layer(struct cnn_Layer* in, struct cnn_Layer *out){
    struct cnn_Layer * newLayer = cnn_create_layer(cnn_SIGMOID_LAYER_NAME, 1, 1, cnn_sigmoid_layer_forward, cnn_sigmoid_layer_backward, cnn_activation_function_layer_initForward, cnn_activation_function_layer_initBackward, 0, 0);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = out;
    return newLayer;
}