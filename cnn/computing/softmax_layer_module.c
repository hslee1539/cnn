#include <math.h>

#include "./softmax_layer_module.h"

void cnn_comput_softmax_crossentropy_layer_forward(struct Tensor* x, struct Tensor* out, int index, int max_index){
    int out0 = index * out->shapes[0] / max_index;
    int out0_max = (index + 1) * out->shapes[0] / max_index;
    float sigma;
    int colMax = out->size / out->shapes[0];
    int pass_index = 0;

    for(; out0 < out0_max; out0++){
        pass_index = out0 * colMax;
        sigma = 0.0f;
        for(int c = 0; c < colMax; c++){
            out->scalas[pass_index + c] = expf(x->scalas[pass_index + c]);
            sigma += out->scalas[pass_index + c];
        }
        for(int c = 0; c < colMax; c++){
            out->scalas[pass_index + c] /= sigma;
        }
    }
}
void cnn_comput_softmax_crossentropy_layer_backward(struct Tensor* dout, struct Tensor* out, struct Tensor* table, struct Tensor* dx, int index, int max_index){
    int dx_index = index * dx->size / max_index;
    int dx_max_index = (index + 1) * dx->size / max_index;

    for(; dx_index < dx_max_index; dx_index++){
        dx->scalas[dx_index] = (out->scalas[dx_index] - table->scalas[dx_index]) * dout->scalas[dx_index / dout->size];
    }

}