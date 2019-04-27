#include <malloc.h>
#include "./updateset_module.h"

struct cnn_UpdateSet* cnn_create_updateset(struct Tensor *delta, struct Tensor *value){
    struct cnn_UpdateSet* newUpdateset = malloc(sizeof(struct cnn_UpdateSet));
    newUpdateset->delta = delta;
    newUpdateset->value = value;
    newUpdateset->momnt = 0;
}
void cnn_release_updateset_deep(struct cnn_UpdateSet *updateset){
    if(updateset->momnt != 0)
        tensor_release_deep(updateset->momnt);
    tensor_release_deep(updateset->delta);
    tensor_release_deep(updateset->value);
}
