#pragma once
#include "./../../tensor/main_module.h"
#include "./updatelist_module.h"
#include "./extradata_module.h"

struct cnn_Optimizer{
    float learning_rate;
    void (*update)(struct cnn_Optimizer *, struct cnn_UpdateList *, int, int);
};

typedef void (*cnn_optimizer_fpUpdate)(struct cnn_Optimizer *, struct cnn_UpdateList *, int, int);

struct cnn_Optimizer*   cnn_create_optimizer    (float learning_rate, cnn_optimizer_fpUpdate update);
void                    cnn_release_optimizer   (struct cnn_Optimizer *optimizer);
void                    cnn_optimizer_update    (struct cnn_Optimizer *optimizer, struct cnn_UpdateList *list, int index, int max_index);
