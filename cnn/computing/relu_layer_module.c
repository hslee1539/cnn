#include "./relu_layer_module.h"

void cnn_comput_relu_layer_forward(struct Tensor* x, struct Tensor* out, int index, int max_index){
    int out_index = out->size * index / max_index;
    int out_max_index = out->size * (index  + 1) / max_index;
    for(; out_index < out_max_index; out_index++){
        out->scalas[out_index] = x->scalas[out_index] * (x->scalas[out_index] > 0);
    }
}
void cnn_comput_relu_layer_backward(struct Tensor* dout, struct Tensor* out, struct Tensor* dx, int index, int max_index){
    int dx_index = out->size * index / max_index;
    int dx_max_index = out->size * (index  + 1) / max_index;
    for(; dx_index < dx_max_index; dx_index++){
        dx->scalas[dx_index] = dout->scalas[dx_index] * (out->scalas > 0);
    }
}