#include "./sigmoid_layer_module.h"

#include "./function/sigmoid_layer_module.h"

char cnn_SIGMOID_LAYER_NAME[] = "sigmoid";

struct cnn_Layer* cnn_create_sigmoid_layer(){
    return cnn_create_layer(cnn_SIGMOID_LAYER_NAME, 0, cnn_sigmoid_layer_forward, cnn_sigmoid_layer_backward, cnn_activation_function_layer_initForward, cnn_activation_function_layer_initBackward, 0, 0, 0);
}