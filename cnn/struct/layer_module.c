#include <malloc.h>
#include "./layer_module.h"

struct cnn_Layer* cnn_create_layer(char *name, int childLayer_size, cnn_layer_callback_computing forward, cnn_layer_callback_computing backward, cnn_layer_callback_init initForward, cnn_layer_callback_init initBackward, cnn_layer_callback_init release, cnn_layer_callback_update update){
    struct cnn_Layer* returnValue = malloc(sizeof(struct cnn_Layer));
    returnValue->name = name;
    returnValue->out = tensor_create();
    returnValue->dx = tensor_create();
    returnValue->inLayer = 0;
    returnValue->outLayer = 0;
    returnValue->childLayer = malloc(sizeof(struct cnn_Layer *) * childLayer_size);
    returnValue->childLayer_size = childLayer_size;
    returnValue->forward = forward;
    returnValue->backward = backward;
    returnValue->initForward = initForward;
    returnValue->initBackward = initBackward;
    returnValue->release = release;
    returnValue->update = update;
    return returnValue;
}
int cnn_release_layer_deep(struct cnn_Layer *layer){
    if(layer->release != 0)
        layer->release(layer);
    tensor_release_deep(layer->out);
    tensor_release_deep(layer->dx);
    for(int i = 0; i < layer->childLayer_size; i++)
        cnn_release_layer_deep(layer->childLayer[i]);
    if(layer->childLayer_size > 0)
        free(layer->childLayer);
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

struct cnn_Layer *cnn_layer_getLeftTerminal(struct cnn_Layer *layer){
    while(layer->childLayer_size > 0)
        layer = layer->childLayer[0];
    return layer;
}
struct cnn_Layer *cnn_layer_getRightTerminal(struct cnn_Layer *layer){
    while(layer->childLayer_size > 0)
        layer = layer->childLayer[layer->childLayer_size - 1];
    return layer;
}
int cnn_layer_link(struct cnn_Layer *left, struct cnn_Layer *right){
    // 양방향 연결 리스트 알고리즘을 참고하세요!
    left = cnn_layer_getRightTerminal(left);
    right = cnn_layer_getLeftTerminal(right);
    left->outLayer = right;
    right->inLayer = left;
    return 0;
}

struct cnn_Layer *cnn_layer_setLearningData(struct cnn_Layer *layer, struct cnn_Layer *data){
    // output
    cnn_layer_getRightTerminal(layer)->outLayer = data;
    // input
    cnn_layer_getLeftTerminal(layer)->inLayer = data;
    return layer;
}