#pragma once
#include "./updateset_module.h"

struct cnn_UpdateList{
    struct cnn_UpdateSet **sets;
    int setSize;
};

struct cnn_UpdateList*  cnn_create_updatelist       (int size);
void                    cnn_release_updatelist_deep (struct cnn_UpdateList* updatelist);