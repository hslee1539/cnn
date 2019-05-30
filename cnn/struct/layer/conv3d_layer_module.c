#include "./conv3d_layer_module.h"
#include "./function/conv3d_layer_module.h"


char cnn_CONV3D_LAYER_NAME[] = "conv3d";

struct cnn_Layer *cnn_create_conv3d_layer(struct cnn_Layer *in, struct cnn_Layer *out, struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding){
    struct cnn_Layer *newLayer = cnn_create_layer(cnn_CONV3D_LAYER_NAME, 1, 1, cnn_conv3d_layer_forward, cnn_conv3d_layer_backward, cnn_conv3d_layer_initForward, cnn_conv3d_layer_initBackward, cnn_conv3d_layer_releaseInnerData, cnn_conv3d_layer_update);
    cnn_conv3d_layer_createInnerData(newLayer, filter, bias, stride, pad, padding);
    newLayer->inLayer[0] = in;
    newLayer->outLayer[0] = out;
    return newLayer;
}
