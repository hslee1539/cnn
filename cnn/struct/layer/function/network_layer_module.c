#include "./network_layer_module.h"

#define CNN_NETWORK_LAYER_INDEX(layer) layer->extra->ints[0]

int cnn_network_layer_createInnerData(struct cnn_Layer* layer){
    layer->extra = cnn_create_extraData(0,0,1);
    CNN_NETWORK_LAYER_INDEX(layer) = 0;
    return 0;
}

int cnn_network_layer_releaseInnerData(struct cnn_Layer* layer){
    cnn_release_extradata(layer->extra);
    return 0;
}

int cnn_network_layer_forward(struct cnn_Layer* layer, int index, int max_index){
    if((index == max_index - 1) && (CNN_NETWORK_LAYER_INDEX(layer) == layer->inLayer_size)){
        CNN_NETWORK_LAYER_INDEX(layer) = 0;
    }
    int layer_index = CNN_NETWORK_LAYER_INDEX(layer);

    int retval = cnn_layer_forward(layer->inLayer[layer_index], index, max_index);

    if(index == max_index - 1){
        CNN_NETWORK_LAYER_INDEX(layer)++;
    }
    return retval + layer->inLayer_size - layer_index - 1;
}

int cnn_network_layer_backward(struct cnn_Layer* layer, int index, int max_index){
    if((index == max_index - 1) && (CNN_NETWORK_LAYER_INDEX(layer) == 0)){
        CNN_NETWORK_LAYER_INDEX(layer) = layer->inLayer_size - 1;
    }
    int layer_index = CNN_NETWORK_LAYER_INDEX(layer);

    int retval = cnn_layer_backward(layer->inLayer[layer_index], index, max_index);

    if(index == max_index - 1)
        CNN_NETWORK_LAYER_INDEX(layer)--;
    return retval + layer_index;
}

int cnn_network_layer_initForward(struct cnn_Layer* layer){
    for(int i = 0; i < layer->inLayer_size; i ++){
        while(cnn_layer_initForward(layer->inLayer[i]));
    }
    
    return 0;
}

int cnn_network_layer_initBackward(struct cnn_Layer* layer){
    for(int i = layer->inLayer_size - 1; i > -1; i--)
        while(cnn_layer_initBackward(layer->inLayer[i]));
    return 0;
}

int cnn_network_layer_update(struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index){
    if((index == max_index - 1) && (CNN_NETWORK_LAYER_INDEX(layer) == layer->inLayer_size)){
        CNN_NETWORK_LAYER_INDEX(layer) = 0;
    }
    int layer_index = CNN_NETWORK_LAYER_INDEX(layer);

    int retval = cnn_layer_update(layer->inLayer[layer_index], optimizer, index, max_index);
    if(index == max_index - 1)
        CNN_NETWORK_LAYER_INDEX(layer)++;
    return retval + layer_index;
}