#pragma once

#include "./tensor.h"

typedef struct ExtraData{
    int tensorSize;
    int floatSize;
    int intSize;
    pTensor *tensors;
    float *floats;
    int *ints;
}ExtraData;

typedef ExtraData *pExtraData;

pExtraData extradata_create(int tensorSize, int floatsSize, int intsSize);
void extradata_release(pExtraData extraData);
void extradata_release_deep(pExtraData extraData);