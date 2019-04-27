#include "./network_layer_module.h"

int cnn_network_create(struct cnn_Layer* layer){
    layer->extra = cnn_create_extraData(0,0,1);
    layer->extra->ints[0] = 0;
    return 0;
}
int cnn_network_release(struct cnn_Layer* layer){
    cnn_release_extradata(layer->extra);
    return 0;
}

int cnn_network_forward(struct cnn_Layer* layer, int index, int max_index){
    int retval = cnn_layer_forward(layer->inLayer[layer->extra->ints[0] % layer->inLayer_size], index, max_index);
    if(index == max_index - 1)
        layer->extra->ints[0]++;
    return retval;
}
int cnn_network_backward(struct cnn_Layer* layer, int index, int max_index){
    int retval = cnn_layer_backward(layer->inLayer[(layer->extra->ints[0] - 1) % layer->inLayer_size], index, max_index);
    if(index == max_index - 1)
        layer->extra->ints[0]--;
    return retval;
}
int cnn_network_initForward(struct cnn_Layer* layer){
    int retval = cnn_layer_initForward(layer->inLayer[layer->extra->ints[0] % layer->inLayer_size]);
    layer->extra->ints[0]++;
    return retval;
}
int cnn_network_initBackward(struct cnn_Layer* layer){
    int retval = cnn_layer_initBackward(layer->inLayer[(layer->extra->ints[0] - 1) % layer->inLayer_size]);
    layer->extra->ints[0]--;
    return retval;
}
int cnn_network_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    int retval = cnn_layer_update(layer->inLayer[layer->extra->ints[0] % layer->inLayer_size], optimizer, index, max_index);
    if(index == max_index - 1)
        layer->extra->ints[0]++;
    return retval;
}