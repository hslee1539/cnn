#include "./batchnorm_layer_module.h"
#include "./../../../computing/batchnorm_layer_module.h"
#include "./standard_layer_define.h"

void cnn_batchnorm_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_batchnorm_layer_forward(CNN_LAYER_X(layer), CNN_BATCHNORM_LAYER_IDSPERSION(layer), CNN_LAYER_OUT(layer), index, max_index);
}
void cnn_batchnorm_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    cnn_comput_batchnorm_layer_backward(CNN_LAYER_DOUT(layer), CNN_LAYER_OUT(layer), CNN_BATCHNORM_LAYER_IDSPERSION(layer), CNN_LAYER_DX(layer), index, max_index);
}
void cnn_batchnorm_layer_initForward(struct cnn_Layer *layer){
    long long size = CNN_LAYER_X(layer)->size / CNN_LAYER_X(layer)->shapes[0];
    if(layer->out->shapes[0] != CNN_LAYER_X(layer)->shapes[0]){
        tensor_release_deep(layer->out);
        layer->out = tensor_create_struct_deep(CNN_LAYER_X(layer));
    }
    if(CNN_BATCHNORM_LAYER_IDSPERSION(layer)->size != size){
        tensor_release_deep(CNN_BATCHNORM_LAYER_IDSPERSION(layer));
        CNN_BATCHNORM_LAYER_IDSPERSION(layer) = tensor_create_values_deep(&size, 1, 0);
    }
}
void cnn_batchnorm_layer_initBackward(struct cnn_Layer *layer){
    long long size = CNN_LAYER_DOUT(layer)->size / CNN_LAYER_DOUT(layer)->shapes[0];
    if(layer->dx->shapes[0] != CNN_LAYER_DOUT(layer)->shapes[0]){
        tensor_release_deep(layer->dx);
        layer->dx = tensor_create_struct_deep(CNN_LAYER_DOUT(layer));
    }
    if(CNN_BATCHNORM_LAYER_IDSPERSION(layer)->size != size){
        tensor_release_deep(CNN_BATCHNORM_LAYER_IDSPERSION(layer));
        CNN_BATCHNORM_LAYER_IDSPERSION(layer) = tensor_create_values_deep(&size, 1, 0);
    }
}

void cnn_batchnorm_layer_createInnerData(struct cnn_Layer *layer){
    layer->extra = cnn_create_extraData(1,0,0);
    layer->extra->tensors[0] = tensor_create();
}
void cnn_batchnorm_layer_releaseInnerData(struct cnn_Layer *layer){
    cnn_release_extradata_deep(layer->extra);
}