#pragma once

#include "./extradata.h"


typedef void(*layer_fpInit)(struct Layer*);
typedef void (*layer_fpPartialComputing)(struct Layer*, int, int);

typedef struct Layer{
    pTensor out;
    pTensor dx;
    pExtraData extra;


    /// 순전파를 분할하여 계산합니다. 해당 레이어, 분할 번호, 분할 수 순으로 인수를 받습니다.
    layer_fpPartialComputing forward;
    /// 역전파를 분할하여 계산합니다. 해당 레이어, 분할 번호, 분할 수 순으로 인수를 받습니다.
    layer_fpPartialComputing backward;
    /// 순전파를 초기화 합니다. 해당 레이어를 인수로 받습니다.
    layer_fpInit initForward;
    /// 역전파를 초기화 합니다. 해당 레이어를 인수로 받습니다.
    layer_fpInit initBackward;
}Layer;

typedef Layer *pLayer;

pLayer layer_create(pTensor out, pTensor dx, pExtraData extra, layer_fpPartialComputing forward, layer_fpPartialComputing backward, layer_fpInit initForward, layer_fpInit initBackward);
void layer_relase_deep(pLayer layer);