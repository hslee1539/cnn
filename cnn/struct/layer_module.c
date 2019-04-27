#include <malloc.h>
#include "./layer_module.h"

struct cnn_Layer* cnn_create_layer(char *name, int inLayer_size, int outLayer_size, cnn_layer_callback_computing forward, cnn_layer_callback_computing backward, cnn_layer_callback_init initForward, cnn_layer_callback_init initBackward, cnn_layer_callback_init release, cnn_layer_callback_update update){
    struct cnn_Layer* returnValue = malloc(sizeof(struct cnn_Layer));
    returnValue->name = name;
    returnValue->out = tensor_create();
    returnValue->dx = tensor_create();
    returnValue->inLayer = malloc(sizeof(struct cnn_Layer*) * inLayer_size);
    returnValue->inLayer_size = inLayer_size;
    returnValue->outLayer = malloc(sizeof(struct cnn_Layer*)* outLayer_size);
    returnValue->outLayer_size = outLayer_size;
    returnValue->forward = forward;
    returnValue->backward = backward;
    returnValue->initForward = initForward;
    returnValue->initBackward = initBackward;
    returnValue->release = release;
    returnValue->update = update;
    return returnValue;
}
int cnn_release_layer_deep(struct cnn_Layer* layer){
    if(layer->release != 0)
        layer->release(layer);
    tensor_release_deep(layer->out);
    tensor_release_deep(layer->dx);
    free(layer);
    return 0;
}

int cnn_layer_forward(struct cnn_Layer *layer, int index, int max_index){
    if(layer->forward)
        return layer->forward(layer, index, max_index);
    return 0;
}
int cnn_layer_backward(struct cnn_Layer *layer, int index, int max_index){
    if(layer->backward)
        return layer->backward(layer, index, max_index);
    return 0;
}
int cnn_layer_initForward(struct cnn_Layer *layer){
    if(layer->initForward)
        return layer->initForward(layer);
    return 0;
}
int cnn_layer_initBackward(struct cnn_Layer *layer){
    if(layer->initBackward)
        return layer->initBackward(layer);
    return 0;
}
int cnn_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    if(layer->update)
        return layer->update(layer, optimizer, index, max_index);
    return 0;
}