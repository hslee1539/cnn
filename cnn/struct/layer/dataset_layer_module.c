#include "./dataset_layer_module.h"

char cnn_DATASET_LAYER_NAME[] = "dataset";



struct cnn_Layer* cnn_create_dataset_layer(struct Tensor* x, struct Tensor* table){
    struct cnn_Layer* newLayer = cnn_create_layer(cnn_DATASET_LAYER_NAME, 0, 0, 0, 0, 0, 0, 0);
    tensor_referTo(newLayer->out, x);
    tensor_referTo(newLayer->dx, table);
    return newLayer;
}
