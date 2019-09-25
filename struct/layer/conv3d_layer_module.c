#include "./conv3d_layer_module.h"
#include "./function/conv3d_layer_module.h"


char cnn_CONV3D_LAYER_NAME[] = "conv3d";

struct cnn_Layer *cnn_create_conv3d_layer(struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding){
    struct cnn_Layer *newLayer = cnn_create_layer(cnn_CONV3D_LAYER_NAME, 0, cnn_conv3d_layer_forward, cnn_conv3d_layer_backward, cnn_conv3d_layer_initForward, cnn_conv3d_layer_initBackward, cnn_conv3d_layer_releaseInnerData, cnn_conv3d_layer_update, _cnn_layer_baseInitUpdate);
    cnn_conv3d_layer_createInnerData(newLayer, filter, bias, stride, pad, padding);
    return newLayer;
}
