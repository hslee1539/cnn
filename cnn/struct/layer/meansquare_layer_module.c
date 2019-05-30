#include "./meansquare_layer_module.h"
#include "./function/meansquare_layer_module.h"

char cnn_MEANSQUARE_LAYER_NAME[] =  "mean square";

struct cnn_Layer *cnn_create_meansquare_layer(struct cnn_Layer *in, struct cnn_Layer *dataset){
    struct cnn_Layer *newLayer = cnn_create_layer(cnn_MEANSQUARE_LAYER_NAME, 1, 1, cnn_meansquare_layer_forward, cnn_meansquare_layer_backward, 0, cnn_activation_function_layer_initBackward, 0, 0);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = dataset;
}