#include "./batchnorm_layer_module.h"
#include "./function/batchnorm_layer_module.h"


char cnn_batchnorm_LAYER_NAME[] = "batchnorm";

struct cnn_Layer *cnn_create_batchnorm_layer(){
    struct cnn_Layer *newLayer = cnn_create_layer(cnn_batchnorm_LAYER_NAME, 0, cnn_batchnorm_layer_forward, cnn_batchnorm_layer_backward, cnn_batchnorm_layer_initForward, cnn_batchnorm_layer_initBackward, cnn_batchnorm_layer_releaseInnerData, 0, 0);
    cnn_batchnorm_layer_createInnerData(newLayer);
    return newLayer;
}
