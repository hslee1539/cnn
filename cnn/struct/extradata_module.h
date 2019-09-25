#pragma once

#include "../tensor/import_module.h"

struct cnn_ExtraData{
    int tensorSize;
    int floatSize;
    int intSize;
    struct Tensor** tensors;
    float *floats;
    int *ints;
};

struct cnn_ExtraData* cnn_create_extraData(int tensorSize, int floatsSize, int intsSize);
void cnn_release_extradata(struct cnn_ExtraData* extraData);
void cnn_release_extradata_deep(struct cnn_ExtraData* extraData);