#include "./softmax_layer_module.h"
#include "./function/softmax_layer_module.h"

char cnn_SOFTMAX_CROSSENTROPY_LAYER_NAME[] = "softmax cross entropy";

struct cnn_Layer *cnn_create_softmax_crossentropy_layer(){
    return cnn_create_layer(cnn_SOFTMAX_CROSSENTROPY_LAYER_NAME, 0, cnn_softmax_crossentropy_layer_forward, cnn_softmax_crossentropy_layer_backward, 0, cnn_activation_function_layer_initBackward, 0, 0, 0);
}