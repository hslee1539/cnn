#include <malloc.h>
#include "./softmax_layer_module.h"

void cnn_softmax_crossentropy_layer_createInnorData(struct cnn_Layer* layer){
    layer->extra = cnn_create_extraData(1,1,0);
    layer->extra->tensors[0] = &tensor_NULL;
    layer->extra->floats = malloc(sizeof(float));
}
void cnn_softmax_crossentropy_layer_releaseInnorData(struct cnn_Layer* layer){
    free(layer->extra->floats);
    cnn_release_extradata(layer->extra);// t는 참조이기 때문에 깊이 메모리를 반환하지 않음.
}

struct Tensor* cnn_softmax_crossentropy_layer_get_table(struct cnn_Layer* layer){
    return layer->extra->tensors[0];
}
void cnn_softmax_crossentropy_layer_set_table(struct cnn_Layer* layer, struct Tensor* table){
    layer->extra->tensors[0] = table;
}

float cnn_softmax_creossentropy_layer_get_loss(struct cnn_Layer* layer){
    return layer->extra->floats[0];
}
void cnn_softmax_crossentropy_layer_set_loss(struct cnn_Layer* layer, float  loss){
    layer->extra->floats[0] = loss;
}