#include "./thread_module.h"
#include "malloc.h" 
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

void *_cnn_thread_thread(void *cnn_ThreadArgs_object){
    cnn_thread_func func;
    int running;
    struct cnn_ThreadArgs *arg = cnn_ThreadArgs_object;
    CNN_SYNCDATA_READ(arg->index, int index);
    printf("%d thread run\n", index);
    CNN_SYNCDATA_WRITE(arg->state, 1);
    CNN_SYNCDATA_READ(arg->running, running);
    while(running){
        printf("%d thread loop!\n", index);
        CNN_SYNCDATA_WRITE(arg->state, 2);
        pthread_mutex_lock(arg->startbody);
            pthread_mutex_lock(arg->start);
                CNN_SYNCDATA_WRITE(arg->state, 3);
            pthread_mutex_unlock(arg->start);
        pthread_mutex_unlock(arg->startbody);
        pthread_mutex_lock(arg->bodybody);
            pthread_mutex_lock(arg->body);
                CNN_SYNCDATA_READ(arg->func, func);
                func(arg);
            pthread_mutex_unlock(arg->body);
        pthread_mutex_unlock(arg->bodybody);
        CNN_SYNCDATA_READ(arg->running, running);
    }
    printf("%d thread die\n", index);
    CNN_SYNCDATA_WRITE(arg->state, 5);
    return 0;
}

void _cnn_thread_forward(struct cnn_ThreadArgs *arg){
    CNN_SYNCDATA_READ(arg->layer, struct cnn_Layer *layer);
    CNN_SYNCDATA_READ(arg->index, int index);
    CNN_SYNCDATA_READ(arg->max, int max_index);
    CNN_SYNCDATA_WRITE(arg->state, 4);
    cnn_layer_forward(layer, index, max_index);
}

void _cnn_thread_backward(struct cnn_ThreadArgs *arg){
    CNN_SYNCDATA_READ(arg->layer, struct cnn_Layer *layer);
    CNN_SYNCDATA_READ(arg->index, int index);
    CNN_SYNCDATA_READ(arg->max, int max_index);
    CNN_SYNCDATA_WRITE(arg->state, 4);
    cnn_layer_backward(layer, index, max_index);
}

void _cnn_thread_update(struct cnn_ThreadArgs *arg){
    CNN_SYNCDATA_READ(arg->layer, struct cnn_Layer *layer);
    CNN_SYNCDATA_READ(arg->index, int index);
    CNN_SYNCDATA_READ(arg->max, int max_index);
    CNN_SYNCDATA_READ(arg->optimizer, struct cnn_Optimizer *optimizer);
    CNN_SYNCDATA_WRITE(arg->state, 4);
    cnn_layer_update(layer, optimizer, index, max_index);
}

struct cnn_Thread *cnn_create_thread(int count){
    struct cnn_Thread *retval = malloc(sizeof(struct cnn_Thread));
    retval->pthread = calloc(sizeof(pthread_t), count);
    retval->args = malloc(sizeof(struct cnn_ThreadArgs) * count);
    retval->count = count;

    retval->starts = malloc(sizeof(pthread_mutex_t) * count);
    retval->startbodys = malloc(sizeof(pthread_mutex_t) * count);
    retval->bodys = malloc(sizeof(pthread_mutex_t) * count);
    retval->bodybodys = malloc(sizeof(pthread_mutex_t) * count);
    
    for(int i = 0; i < count; i++){
        pthread_mutex_init(retval->starts + i, 0);
        pthread_mutex_init(retval->startbodys + i, 0);
        pthread_mutex_init(retval->bodys + i, 0);
        pthread_mutex_init(retval->bodybodys + i, 0);

        pthread_mutex_lock(retval->starts + i);

        (retval->args + i)->start = retval->starts + i;
        (retval->args + i)->startbody = retval->startbodys + i;
        (retval->args + i)->body = retval->bodys + i;
        (retval->args + i)->bodybody = retval->bodybodys + i;

        CNN_CREATE_SYNCDATA(struct cnn_SyncIndex, (retval->args + i)->index);
        CNN_CREATE_SYNCDATA(struct cnn_SyncIndex, (retval->args + i)->max);
        CNN_CREATE_SYNCDATA(struct cnn_SyncState, (retval->args + i)->state);

        CNN_CREATE_SYNCDATA(struct cnn_SyncFunc, (retval->args + i)->func);
        CNN_CREATE_SYNCDATA(struct cnn_SyncLayer, (retval->args + i)->layer);
        CNN_CREATE_SYNCDATA(struct cnn_SyncOptimizer, (retval->args + i)->optimizer);
        CNN_CREATE_SYNCDATA(struct cnn_SyncRunning, (retval->args + i)->running);

        CNN_SYNCDATA_WRITE((retval->args + i)->index, i);
        CNN_SYNCDATA_WRITE((retval->args + i)->max, count);
        CNN_SYNCDATA_WRITE((retval->args + i)->state, 0);
    }
    return retval;
}

void cnn_release_thread(struct cnn_Thread *thread){
    for(int i = 0; i < thread->count; i++){
        CNN_SYNCDATA_READ((thread->args + i)->state, int state);
        if(state % 5){
            CNN_SYNCDATA_WRITE((thread->args + i)->running, 0);
            pthread_mutex_unlock(thread->starts + i);
            pthread_join(*(thread->pthread + i), 0);
        }

        CNN_RELEASE_SUNCDATA((thread->args + i)->func);
        CNN_RELEASE_SUNCDATA((thread->args + i)->index);
        CNN_RELEASE_SUNCDATA((thread->args + i)->layer);
        CNN_RELEASE_SUNCDATA((thread->args + i)->max);
        CNN_RELEASE_SUNCDATA((thread->args + i)->optimizer);
        CNN_RELEASE_SUNCDATA((thread->args + i)->running);
        CNN_RELEASE_SUNCDATA((thread->args + i)->state);
    }
    free(thread->starts);
    free(thread->bodys);
    free(thread->pthread);
    free(thread->args);
    free(thread);
}

void cnn_thread_forward(struct cnn_Thread *thread, struct cnn_Layer *layer){
    printf("main thread forward\n");
    int index;
    // 모든 이전 작업 완료 대기
    for(index = 0; index < thread->count; index++){
        // 쓰래드가 body를 사용완료까지 대기
        pthread_mutex_lock(thread->bodybodys + index);
        pthread_mutex_unlock(thread->bodybodys + index);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->bodys + index);
        pthread_mutex_unlock(thread->starts + index);
        CNN_SYNCDATA_WRITE((thread->args + index)->func, _cnn_thread_forward);
        CNN_SYNCDATA_WRITE((thread->args + index)->layer, layer);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->startbodys + index);
        pthread_mutex_unlock(thread->startbodys + index);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_unlock(thread->bodys + index);
        pthread_mutex_lock(thread->starts + index);
    }
}
void cnn_thread_backward(struct cnn_Thread *thread, struct cnn_Layer *layer){
    printf("main thread backward\n");
    int index;
    // 모든 이전 작업 완료 대기
    for(index = 0; index < thread->count; index++){
        // 쓰래드가 body를 사용완료까지 대기
        pthread_mutex_lock(thread->bodybodys + index);
        pthread_mutex_unlock(thread->bodybodys + index);
    }
    // 여기까지 왔으면
    // 모든 쓰래드는 start 부분에 정지됨.
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->bodys + index);
        pthread_mutex_unlock(thread->starts + index);
        CNN_SYNCDATA_WRITE((thread->args + index)->func, _cnn_thread_backward);
        CNN_SYNCDATA_WRITE((thread->args + index)->layer, layer);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->startbodys + index);
        pthread_mutex_unlock(thread->startbodys + index);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_unlock(thread->bodys + index);
        pthread_mutex_lock(thread->starts + index);
    }
}

void cnn_thread_update(struct cnn_Thread *thread, struct cnn_Layer *layer, struct cnn_Optimizer *optimizer){
    printf("main thread update\n");
    int index;
    // 모든 이전 작업 완료 대기
    for(index = 0; index < thread->count; index++){
        // 쓰래드가 body를 사용완료까지 대기
        pthread_mutex_lock(thread->bodybodys + index);
        pthread_mutex_unlock(thread->bodybodys + index);
    }
    // 여기까지 왔으면
    // 모든 쓰래드는 start 부분에 정지됨.
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->bodys + index);
        pthread_mutex_unlock(thread->starts + index);
        CNN_SYNCDATA_WRITE((thread->args + index)->func, _cnn_thread_update);
        CNN_SYNCDATA_WRITE((thread->args + index)->layer, layer);
        CNN_SYNCDATA_WRITE((thread->args + index)->optimizer, optimizer);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_lock(thread->startbodys + index);
        pthread_mutex_unlock(thread->startbodys + index);
    }
    for(index = 0; index < thread->count; index++){
        pthread_mutex_unlock(thread->bodys + index);
        pthread_mutex_lock(thread->starts + index);
    }
}

int cnn_thread_networkNext(struct cnn_Thread *thread, struct cnn_Layer *layer){
    printf("main thread networkNext\n");
    int index;
    // 모든 이전 작업 완료 대기
    for(index = 0; index < thread->count; index++){
        // 쓰래드가 body를 사용완료까지 대기
        //CNN_SYNCDATA_READ((thread->args + index)->state, int state);
        pthread_mutex_lock(thread->bodybodys + index);
        pthread_mutex_unlock(thread->bodybodys + index);
        //printf("%d\n", state);
    }
    
    return cnn_network_next(layer);
}

void cnn_thread_start(struct cnn_Thread *thread){
    for(int i = 0; i < thread->count; i++){
        // 쓰래드 반복문 정지
        CNN_SYNCDATA_WRITE((thread->args + i)->running, 0);
        // 시작 제어 동기화 해제
        pthread_mutex_unlock(thread->starts + i);
        // 끝날 때 까지 대기
        pthread_join(*(thread->pthread + i), 0);

        // 쓰래드 반복문 무한 반복하게 하기
        CNN_SYNCDATA_WRITE((thread->args + i)->running, 1);
        // 하지만 반복문 정지
        pthread_mutex_lock(thread->starts + i);
        // 생성
        pthread_create(thread->pthread + i, 0, _cnn_thread_thread, thread->args + i);
    }
}
void cnn_thread_end(struct cnn_Thread *thread){
    for(int i = 0; i < thread->count; i++){
        // 쓰래드 반복문 정지
        CNN_SYNCDATA_WRITE((thread->args + i)->running, 0);
        pthread_mutex_unlock(thread->starts + i);
        pthread_join(*(thread->pthread + i), 0);
    }
}