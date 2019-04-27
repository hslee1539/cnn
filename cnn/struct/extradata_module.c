#include <malloc.h>
#include "./extradata_module.h"

struct cnn_ExtraData* cnn_create_extraData(int tensorSize, int floatsSize, int intsSize){
    struct cnn_ExtraData* out = malloc(sizeof(struct cnn_ExtraData));
    if(floatsSize > 0)
        out->floats = malloc(sizeof(float) * floatsSize);
    if(intsSize > 0)
        out->ints = malloc(sizeof(int) * intsSize);
    if(tensorSize > 0)
        out->tensors = malloc(sizeof(struct Tensor*) * tensorSize);
    out->floatSize = floatsSize;
    out->intSize = intsSize;
    out->tensorSize = tensorSize;
    return out;
}
void cnn_release_extradata(struct cnn_ExtraData* extraData){
    if(extraData->floatSize > 0)
        free(extraData->floats);
    if(extraData->intSize > 0)
        free(extraData->ints);
    if(extraData->tensorSize > 0)
        free(extraData->tensors);
    free(extraData);
}
void cnn_release_extradata_deep(struct cnn_ExtraData* extraData){
    for(int i = 0; i < extraData->tensorSize; i++){
        tensor_release_deep(extraData->tensors[i]);
    }
    if(extraData->floatSize > 0)
        free(extraData->floats);
    if(extraData->intSize > 0)
        free(extraData->ints);
    if(extraData->tensorSize > 0)
        free(extraData->tensors);
    free(extraData);
}