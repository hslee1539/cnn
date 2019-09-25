#pragma once
#include "../tensor/import_module.h"

struct cnn_UpdateSet{
    struct Tensor *delta;
    struct Tensor *value;
    struct Tensor *momnt;
};

struct cnn_UpdateSet*   cnn_create_updateset        (struct Tensor *delta, struct Tensor *value);
void                    cnn_release_updateset_deep  (struct cnn_UpdateSet *updateset);