#include "./deconv3d_layer_module.h"
#include "./function/deconv3d_layer_module.h"


char cnn_DECONV3D_LAYER_NAME[] = "deconv3d";

struct cnn_Layer *cnn_create_deconv3d_layer(struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding){
    struct cnn_Layer *newLayer = cnn_create_layer(cnn_DECONV3D_LAYER_NAME, 0, cnn_deconv3d_layer_forward, cnn_deconv3d_layer_backward, cnn_deconv3d_layer_initForward, cnn_deconv3d_layer_initBackward, cnn_deconv3d_layer_releaseInnerData, cnn_deconv3d_layer_update);
    cnn_deconv3d_layer_createInnerData(newLayer, filter, bias, stride, pad, padding);
    return newLayer;
}
