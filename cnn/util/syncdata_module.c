#include "./syncdata_module.h"
#include <malloc.h>
/* 
struct cnn_SyncData *cnn_create_syncData(int data){
    struct cnn_SyncData *retval = malloc(sizeof(struct cnn_SyncData));
    retval->data = data;
    pthread_mutex_init(&retval->io_mutex, 0);
    return retval;
}
void cnn_release_syncData(struct cnn_SyncData *syncData){
    pthread_mutex_destroy(&syncData->io_mutex);
    free(syncData);
}

int cnn_syncData_syncRead(struct cnn_SyncData *syncData){
    int retval;
    pthread_mutex_lock(&syncData->io_mutex);
    retval = syncData->data;
    pthread_mutex_unlock(&syncData->io_mutex);
    return retval;
}
void cnn_syncData_syncWrite(struct cnn_SyncData *syncData, int data){
    pthread_mutex_lock(&syncData->io_mutex);
    syncData->data = data;
    pthread_mutex_unlock(&syncData->io_mutex);
}
*/