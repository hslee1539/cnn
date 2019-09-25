#include "./sigmoid_layer_module.h"
#include <math.h>

void cnn_comput_sigmoid_layer_forward(struct Tensor* x, struct Tensor* out, int index, int max_index){
    for(int i = index * x->size / max_index, i_max = (index + 1) * x->size / max_index; i < i_max; i ++){
        //for i in range(index * len(x_array) // max_index, (index + 1) * len(x_array) // max_index):
        out->scalas[i] = 1 / (1 + expf(-x->scalas[i]));
        //out_array[i] = 1 / (1 + math.exp(-x_array[i]))
    }
}

void cnn_comput_sigmoid_layer_backward(struct Tensor* dout, struct Tensor* out, struct Tensor* dx, int index, int max_index){
    for(int i = index * dout->size / max_index, i_max = (index + 1) * dout->size / max_index; i < i_max; i ++){
        //for i in range(index * len(dout_array) // max_index, (index + 1) * len(dout_array) // max_index):
        dx->scalas[i] = dout->scalas[i] * (1 - out->scalas[i]) * out->scalas[i];
        //out_array[i] = dout_array[i] * (1 - out_array[i]) * out_array[i]
    }
}
