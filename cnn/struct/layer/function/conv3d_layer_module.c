#include "./conv3d_layer_module.h"
#include "./../../../computing/conv3d_layer_module.h"
#include "./standard_updatable_layer_define.h"

#define CNN_CONV3D_LAYER_STRIDE(layer) layer->extra->ints[0]
#define CNN_CONV3D_LAYER_PAD(layer) layer->extra->ints[1]
#define CNN_CONV3D_LAYER_PADDING(layer) layer->extra->ints[2]

int cnn_conv3d_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    cnn_comput_conv3d_layer_dfilter(CNN_LAYER_DOUT(layer), CNN_LAYER_X(layer), CNN_CONV3D_LAYER_STRIDE(layer), CNN_CONV3D_LAYER_PAD(layer), CNN_CONV3D_LAYER_PADDING(layer), CNN_LAYER_FILTER(layer)->delta, index, max_index);
    cnn_comput_conv3d_layer_dbias(CNN_LAYER_DOUT(layer), CNN_LAYER_B(layer)->delta, index, max_index);
    cnn_optimizer_update(optimizer, layer->updateList, index, max_index);
    return 0;
}
int cnn_conv3d_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_conv3d_layer_forward(CNN_LAYER_X(layer), CNN_LAYER_FILTER(layer)->value, CNN_LAYER_B(layer)->value, CNN_CONV3D_LAYER_STRIDE(layer), CNN_CONV3D_LAYER_PAD(layer), CNN_CONV3D_LAYER_PADDING(layer), layer->out, index, max_index);
    return 0;
}
int cnn_conv3d_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_conv3d_layer_backward(CNN_LAYER_DOUT(layer), CNN_LAYER_FILTER(layer)->value, CNN_CONV3D_LAYER_STRIDE(layer), CNN_CONV3D_LAYER_PAD(layer), layer->dx, index, max_index);
    return 0;
}
int cnn_conv3d_layer_initForward(struct cnn_Layer *layer){
    if(layer->out->shapes[0] != CNN_LAYER_X(layer)->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = cnn_create_conv3d_layer_out(CNN_LAYER_X(layer), CNN_LAYER_FILTER(layer)->value, CNN_CONV3D_LAYER_STRIDE(layer), CNN_CONV3D_LAYER_PAD(layer));
    }
    return 0;
}
int cnn_conv3d_layer_initBackward(struct cnn_Layer *layer){
    if(layer->dx->shapes[0] != CNN_LAYER_DOUT(layer)->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(CNN_LAYER_X(layer));
    }
    return 0;
}

int cnn_conv3d_layer_createInnerData(struct cnn_Layer *layer, struct Tensor *filter, struct Tensor *bias, int stride, int pad, int padding){
    layer->updateList = cnn_create_updatelist(2);
    CNN_LAYER_FILTER(layer) = cnn_create_updateset(tensor_create_struct_deep(filter), filter);
    CNN_LAYER_B(layer) = cnn_create_updateset(tensor_create_struct_deep(bias), bias);
    layer->extra = cnn_create_extraData(0,0,3);
    CNN_CONV3D_LAYER_STRIDE(layer) = stride;
    CNN_CONV3D_LAYER_PAD(layer) = pad;
    CNN_CONV3D_LAYER_PADDING(layer) = padding;
    return 0;
}

int cnn_conv3d_layer_releaseInnerData(struct cnn_Layer *layer){
    cnn_release_updatelist_deep(layer->updateList);
    cnn_release_extradata_deep(layer->extra);
    return 0;
}

