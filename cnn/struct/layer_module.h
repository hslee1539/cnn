#pragma once

#include "./optimizer_module.h"

///레이어 구조체입니다.
struct cnn_Layer;

typedef void         (*cnn_layer_callback_init)      (struct cnn_Layer*);
typedef void         (*cnn_layer_callback_computing) (struct cnn_Layer*, int, int);
typedef void         (*cnn_layer_callback_update)    (struct cnn_Layer*, struct cnn_Optimizer*, int, int);

struct cnn_Layer{
    /// 레이어의 이름입니다. 레이어의 이름은 중복이 없는 것을 원칙으로 합니다.
    char                    *name;
    /// 이 레이어의 순전파 결과입니다.
    struct Tensor           *out;
    /// 이 레이어의 역전파 결과입니다.
    struct Tensor           *dx;

    // TODO: 정리하자!
    struct cnn_UpdateList   *updateList;
    struct cnn_ExtraData    *extra;
    
    struct cnn_Layer        *inLayer;
    struct cnn_Layer        *outLayer;
    struct cnn_Layer        **childLayer;
    int                     childLayer_size;


    /// 순전파를 할때 호출되니다. 해당 레이어, 분할 번호, 분할 수 순으로 인수를 받습니다.
    cnn_layer_callback_computing    forward;
    /// 역전파를 할때 호출됩니다. 해당 레이어, 분할 번호, 분할 수 순으로 인수를 받습니다.
    cnn_layer_callback_computing    backward;
    /// 순전파를 초기화할때 호출됩니다.
    cnn_layer_callback_init         initForward;
    /// 역전파를 초기화할때 호출됩니다.
    cnn_layer_callback_init         initBackward;
    /// 레이어를 해제할때 호출됩니다.
    cnn_layer_callback_init         release;
    /// 레이어가 업데이트 할때 호출됩니다.
    cnn_layer_callback_update       update;
};

///신경망을 만듭니다.
struct cnn_Layer    *cnn_create_layer               (char *name, int childLayer_size, cnn_layer_callback_computing forward, cnn_layer_callback_computing backward, cnn_layer_callback_init initForward, cnn_layer_callback_init initBackward, cnn_layer_callback_init release, cnn_layer_callback_update update);

///레이어를 깊이 반환합니다.
int                 cnn_release_layer_deep          (struct cnn_Layer *layer);


struct cnn_Layer *cnn_layer_forward               (struct cnn_Layer *layer, int index, int max_index);
struct cnn_Layer *cnn_layer_backward              (struct cnn_Layer *layer, int index, int max_index);
struct cnn_Layer *cnn_layer_initForward           (struct cnn_Layer *layer);
struct cnn_Layer *cnn_layer_initBackward          (struct cnn_Layer *layer);
struct cnn_Layer *cnn_layer_update                (struct cnn_Layer *layer, struct cnn_Optimizer *optimizer, int index, int max_index);
/// 레이어의 왼쪽 터미널(말단) 레이어를 반환합니다.
/// 만약 이 레이어가 말단 레이어면, 이 레이어를 반환합니다.
struct cnn_Layer    *cnn_layer_getLeftTerminal      (struct cnn_Layer *layer);
/// 레이어의 오른쪽 터미널(말단) 레이어를 반환합니다.
/// 만약 이 레이어가 말단 레이어면, 이 레이어를 반환합니다.
struct cnn_Layer    *cnn_layer_getRightTerminal     (struct cnn_Layer *layer);
/// 두개의 레이어를 참조 연결합니다.
/// left가 input쪽이고, right이 output쪽입니다.
int                 cnn_layer_link                  (struct cnn_Layer *left, struct cnn_Layer *right);
/// layer에 학습 데이터를 설정합니다.
/// layer의 왼쪽 터미널 레이어의 inLayer로 데이터 레이어가 참조되고, 오른쪽 터미널의 outLayer에 데이터 레이어가 참조됩니다.
struct cnn_Layer    *cnn_layer_setLearningData      (struct cnn_Layer *layer, struct cnn_Layer *data);