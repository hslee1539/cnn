#pragma once
#include "./../struct/layer_module.h"
#include "./../struct/layer/network_layer_module.h"
#include "./../struct/optimizer_module.h"
#include "./syncdata_module.h"
#include <pthread.h>

struct cnn_ThreadArgs;
typedef void (*cnn_thread_func)(struct cnn_ThreadArgs *);

// 이렇게 구현한 이유는 흐름 제어 동기화랑 각 제어 변수 동기화랑 구별되게 하기 위해서임.
CNN_TYPEDEF_SYNCDATA(cnn_SyncState, int);
CNN_TYPEDEF_SYNCDATA(cnn_SyncRunning, int);
CNN_TYPEDEF_SYNCDATA(cnn_SyncIndex, int);
CNN_TYPEDEF_SYNCDATA(cnn_SyncLayer, struct cnn_Layer *);
CNN_TYPEDEF_SYNCDATA(cnn_SyncOptimizer, struct cnn_Optimizer *);
CNN_TYPEDEF_SYNCDATA(cnn_SyncFunc, cnn_thread_func);

struct cnn_ThreadArgs{
    // 흐름 제어 동기화#############################################################################################

    /// 쓰래드에서 cnn_ThreadArgs::func 를 처리하기 전, lock, unlock이 되는 부분입니다.
    pthread_mutex_t *start;
    /// 쓰래드에서 start에 lock하기전에 lock을 하고, start를 unlock하면 똑같이 unlock합니다.
    pthread_mutex_t *startbody;
    /// 쓰래드에서 cnn_ThreadArgs::func 를 처리하는 부분입니다.
    pthread_mutex_t *body;
    pthread_mutex_t *bodybody;
    
    // 메인 쓰래드만 write 하는 sync 구조체들########################################################################
        
    struct cnn_SyncFunc *func;
    struct cnn_SyncLayer *layer;
    struct cnn_SyncOptimizer *optimizer; 
    struct cnn_SyncIndex *index; // 이 부분은 동기화가 필요 없을 지도 모름
    struct cnn_SyncIndex *max; // 이 부분은 동기화가 필요 없을 지도 모름
    struct cnn_SyncRunning *running;

    // 이 쓰래드가 write하는 sync 구조체들###########################################################################

    /// cnn_thread_thread 함수에서 이 쓰래드의 상태를 나타냅니다. 이 함수 외에는 읽기만 하세요.
    /// 0 : 쓰래드 첫 생성 전 상태, 1 : 쓰래드 초기 상태, 2 : 뮤택스 start 상태, 3 : 뮤택스 body 상태, 4 : 뮤택스 , 5 : 종료
    struct cnn_SyncState *state;
};

struct cnn_Thread{
    // 흐름 제어 동기화#############################################################################################

    pthread_mutex_t *starts;
    pthread_mutex_t *startbodys;
    pthread_mutex_t *bodys;
    pthread_mutex_t *bodybodys;
    
    // 변수들######################################################################################################
    /// 쓰래드 아이디
    pthread_t *pthread;
    /// 쓰래드에 하나씩 할당되는 변수
    struct cnn_ThreadArgs *args;
    /// 쓰래드 수 ( = args, pthread의 수)
    int count;
};


struct cnn_Thread *cnn_create_thread(int count);
void cnn_release_thread(struct cnn_Thread *thread);

/// 멀티 쓰래드로 해당 레이어를 비동기 순전파를 합니다. 만약 이전에 처리중인 일이 있으면 처리중인 일이 끝날 때 까지 정지됩니다.
void cnn_thread_forward(struct cnn_Thread *thread, struct cnn_Layer *layer);
/// 멀티 쓰래드로 해당 레이어를 비동기 역전파를 합니다. 만약 이전에 처리중인 일이 있으면 처리중인 일이 끝날 때 까지 정지됩니다.
void cnn_thread_backward(struct cnn_Thread *thread, struct cnn_Layer *layer);
/// 멀티 쓰래드로 해당 레이어를 비동기 업데이트 합니다. 만약 이전에 처리중인 일이 있으면 처리중인 일이 끝날 때 까지 정지됩니다.
void cnn_thread_update(struct cnn_Thread *thread, struct cnn_Layer *layer, struct cnn_Optimizer *optimizer);
/// 메인 쓰래드로 네트워크가 다음 레이어에 대한 연산을 수행하기 위한 연산을 합니다. 만약 처리중인 
int cnn_thread_networkNext(struct cnn_Thread *thread, struct cnn_Layer *layer);

/// 쓰래드를 만들고 시작합니다. 재시작도 가능합니다.
void cnn_thread_start(struct cnn_Thread *thread);
/// 쓰래드를 종료 신호를 보내고 기다립니다.
void cnn_thread_end(struct cnn_Thread *thread);