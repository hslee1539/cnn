#include "./pointer_module.h"

unsigned long long cnn_getLayerAdress(struct cnn_Layer *layer){
    return (unsigned long long)layer;
}