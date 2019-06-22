#pragma once
#include "./layer_module.h"

struct cnn_Iterator{
    struct cnn_Layer *start;
    struct cnn_Layer *stop;
    struct cnn_Layer *cur;
};

struct cnn_Iterator *cnn_create_iterator1(struct cnn_Layer *network);
struct cnn_Iterator *cnn_create_iterator2(struct cnn_Layer *start, struct cnn_Layer *stop);
void cnn_release_iterator(struct cnn_Iterator *iter);

/// 다음 레이어로 이동합니다. 끝인 경우, 0 포인터를 반환합니다.
struct cnn_Layer *cnn_iterator_next(struct cnn_Iterator *iter);
// 이전 레이어로 이동합니다. 끝인 경우, 0포인터를 반환합니다.
struct cnn_Layer *cnn_iterator_back(struct cnn_Iterator *iter);