#include "./meansquare_layer_module.h"

#define JEGOP(x) x * x

void cnn_comput_meansquare_layer_forward(struct Tensor* x, struct Tensor* table, struct Tensor* out, int index, int max_index){
    float tmp = 0;
    for(int i = index * x->size / max_index, i_max = (index + 1) * x->size / max_index; i < i_max; i ++){
        tmp += JEGOP(table->scalas[i] - x->scalas[i]);
    }
    // 멀티 쓰래드를 지원시 충돌날 위험이 있음.
    out->scalas[0] += tmp / 2;
    // loss값을 저장되는 out의 size를 늘려서 회피하는 방법
    //out->scalas[index % out->size] += tmp / 2;
}

void cnn_comput_meansquare_layer_backward(struct Tensor* x, struct Tensor* table, struct Tensor* dx, int index, int max_index){
    float tmp = 0;
    for(int i = index * dx->size / max_index, i_max = (index + 1) * dx->size / max_index; i < i_max; i ++){
        dx->scalas[i] = table->scalas[i] - x->scalas[i];
    }
}