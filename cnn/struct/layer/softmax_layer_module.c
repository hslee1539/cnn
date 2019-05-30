#include "./softmax_layer_module.h"
#include "./function/softmax_layer_module.h"

char cnn_SOFTMAX_CROSSENTROPY_LAYER_NAME[] = "softmax cross entropy";

struct cnn_Layer *cnn_create_softmax_crossentropy_layer(struct cnn_Layer *in, struct cnn_Layer *dataset){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_SOFTMAX_CROSSENTROPY_LAYER_NAME, 1, 1, cnn_softmax_crossentropy_layer_forward, cnn_softmax_crossentropy_layer_backward, 0, cnn_activation_function_layer_initBackward, 0, 0);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = dataset;
    return newLayer;
}