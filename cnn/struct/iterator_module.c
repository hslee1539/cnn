#include "./iterator_module.h"
#include <malloc.h>

struct cnn_Iterator *cnn_create_iterator1(struct cnn_Layer *network){
    struct cnn_Iterator *retval = malloc(sizeof(struct cnn_Iterator));
    retval->cur = 0;
    retval->start = cnn_layer_getLeftTerminal(network);
    retval->stop = cnn_layer_getRightTerminal(network);
    return retval;
}

struct cnn_Iterator *cnn_create_iterator2(struct cnn_Layer *start, struct cnn_Layer *stop){
    struct cnn_Iterator *retval = malloc(sizeof(struct cnn_Iterator));
    retval->cur = 0;
    retval->start = start;
    retval->stop = stop;
    return retval;
}

void cnn_release_iterator(struct cnn_Iterator *iter){
    free(iter);
}

struct cnn_Layer *cnn_iterator_next(struct cnn_Iterator *iter){
    if(iter->cur == 0){
        iter->cur = iter->start;
    }
    else if(iter->cur == iter->stop){
        iter->cur = 0;
    }
    else{
        iter->cur = iter->cur->outLayer;
    }
    return iter->cur;
}

struct cnn_Layer *cnn_iterator_back(struct cnn_Iterator *iter){
    if(iter->cur == 0){
        iter->cur = iter->stop;
    }
    else if(iter->cur == iter->start){
        iter->cur = 0;
    }
    else{
        iter->cur = iter->cur->inLayer;
    }
    return iter->cur;
}