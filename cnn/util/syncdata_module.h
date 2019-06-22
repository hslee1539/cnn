#pragma once
#include <pthread.h>
#include <malloc.h>

#define CNN_SYNCDATA_READ(syncData_object, returnValue) pthread_mutex_lock(&((syncData_object)->io_mutex));\
returnValue = (syncData_object)->data;\
pthread_mutex_unlock(&((syncData_object)->io_mutex))
                                                

#define CNN_SYNCDATA_WRITE(syncData_object, value) pthread_mutex_lock(&((syncData_object)->io_mutex));\
(syncData_object)->data = value;\
pthread_mutex_unlock(&((syncData_object)->io_mutex))

#define CNN_TYPEDEF_SYNCDATA(structName, type) struct structName{\
type data;\
pthread_mutex_t io_mutex;\
}

#define CNN_CREATE_SYNCDATA(structName, objectName) objectName = malloc(sizeof( structName ));\
pthread_mutex_init(&((objectName)->io_mutex), 0)

#define CNN_RELEASE_SUNCDATA(objectName) pthread_mutex_destroy(&((objectName)->io_mutex)); free((objectName))