#include <malloc.h>
#include "./updatelist_module.h"

struct cnn_UpdateList* cnn_create_updatelist(int size){
    struct cnn_UpdateList* newUpdateList = malloc(sizeof(struct cnn_UpdateList));
    newUpdateList->setSize = size;
    if(size > 0)
        newUpdateList->sets = malloc(sizeof(struct cnn_UpdateSet) * size);
    return newUpdateList;
}
void cnn_release_updatelist_deep(struct cnn_UpdateList* updatelist){
    for(int i = 0; i < updatelist->setSize; i++){
        cnn_release_updateset_deep(updatelist->sets[i]);
    }
    if(updatelist->setSize > 0){
        free(updatelist->sets);
    }
    free(updatelist);
}