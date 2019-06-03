#include "./meansquare_layer_module.h"
#include "./function/meansquare_layer_module.h"

char cnn_MEANSQUARE_LAYER_NAME[] =  "mean square";

struct cnn_Layer *cnn_create_meansquare_layer(){
    return cnn_create_layer(cnn_MEANSQUARE_LAYER_NAME, 0, cnn_meansquare_layer_forward, cnn_meansquare_layer_backward, 0, cnn_meansquare_layer_initBackward, 0, 0);
}