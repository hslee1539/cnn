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
int cnn_release_layer_deep(struct cnn_Layer *layer){
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

int cnn_layer_link(struct cnn_Layer *layer, struct cnn_Layer *source){
    struct cnn_Layer *target1 = layer;
    struct cnn_Layer *target2 = layer;
    // dataset은 0개이고, 나머지는 1개이고, 네트워크는 0개 이상을 가질 수 있음.
    // 즉 inLayer_size가 2개 이상일 경우, layer->inLayer에 해당되는 레이어들과 layer와의 관계는 수평적 레이어 관계가 아닌 수직적 레이어 관계임.
    while(target1->inLayer_size > 1){
        target1 = target1->inLayer[0];
    }
    while(target2->outLayer_size > 1){
        target2 = target2->outLayer[target2->outLayer_size - 1];
    }
    target1->inLayer[0] = source
    target2->outLayer[target2->outLayer_size - 1] = source;
}