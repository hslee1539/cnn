#include "./fully_connected_layer_module.h"
#include <malloc.h>

/// 완전연결층의 연산결과를 저장할 tensor를 생성합니다.
struct Tensor* cnn_create_fully_connected_out(struct Tensor* x, struct Tensor* w){
    long long *tmpShapes = malloc(sizeof(long long) * w->dim);
    tmpShapes[0] = x->shapes[0];
    for(int i = 1; i < w->dim; i++){
        tmpShapes[i] = w->shapes[i];
    }
    struct Tensor *returnValue = tensor_create_values_deep(tmpShapes, w->dim, 0.0f);
    free(tmpShapes);
    return returnValue;
}

/*
void cnn_comput_fully_connected_relu_layer_forward(struct Tensor* x, struct Tensor* w, struct Tensor* b, struct Tensor* out, int index, int max_index){
    int out_index = index * out->size / max_index;
    int out_index_max = (index + 1) * out->size / max_index;
    int product_index = 0;
    int pass_x_index = 0;
    int pass_w_index = 0;
    float tmp = 0.0f;
    for (; out_index < out_index_max; out_index++){
        pass_x_index = out_index / b->shapes[0] * w->shapes[0];
        pass_w_index = out_index % b->shapes[0];

        tmp = 0;
        for(product_index = 0; product_index < w->shapes[0]; product_index++)
            tmp += x->scalas[pass_x_index + product_index] * w->scalas[pass_w_index + b->shapes[0] * product_index];
        tmp += b->scalas[pass_w_index];
        out->scalas[out_index] = tmp * (tmp > 0);
    }
}
// 기존 relu의 forward 결과의 out이 손실되고 relu의 backward의 dx로 갱신이 됨.
void cnn_comput_fully_connected_relu_layer_backward(struct Tensor* dout, struct Tensor* w, struct Tensor* out, struct Tensor* activation_dx,struct Tensor* dx, int index, int max_index){
    int dx_index = index * dx->size / max_index;
    int dx_index_max = (index + 1) * dx->size / max_index;
    int product_index = 0;
    int pass_dout_index = 0;
    int pass_w_index = 0;

    float tmp = 0.0f;

    for (; dx_index < dx_index_max; dx_index++){
        pass_dout_index = dx_index / w->shapes[0] * w->shapes[1];
        pass_w_index = dx_index % w->shapes[0] * w->shapes[1];

        tmp = 0;
        for (product_index = 0; product_index < w->shapes[1]; product_index++){
            activation_dx->scalas[pass_dout_index + product_index] = (out->scalas[pass_dout_index + product_index] > 0) * dout->scalas[pass_dout_index + product_index];
            tmp += activation_dx->scalas[pass_dout_index + product_index] * w->scalas[pass_w_index + product_index];
        }
        dx->scalas[dx_index] = tmp;
    }
}

void cnn_comput_fully_connected_relu_layer_dw(struct Tensor* x, struct Tensor* activation_dx, struct Tensor* dw, int index, int max_index){
    int dw_index = index * dw->size / max_index;
    int dw_index_max = (index  + 1) * dw->size / max_index;
    int dw1 = 0;
    int dw0 = 0;
    int x0 = 0;
    float tmp = 0.0f;

    for(; dw_index < dw_index_max; dw_index++){
        dw1 = dw_index % dw->shapes[1];
        dw0 = dw_index / dw->shapes[1];
        tmp = 0;
        for(x0 = 0; x0 < x->shapes[0]; x0++)
            tmp += x->scalas[x0 * x->shapes[1] + dw0] * activation_dx->scalas[x0 * dw->shapes[1] + dw1];
        dw->scalas[dw_index] = tmp;
    }
}

void cnn_comput_fully_connected_relu_layer_db(struct Tensor* activation_dx, struct Tensor* db, int index, int max_index){
    int db_index = index * db->size / max_index;
    int db_index_max = (index + 1) * db->size / max_index;
    int d0 = 0;
    float tmp = 0.0f;

    for(; db_index < db_index_max; db_index++){
        tmp = 0;
        for(d0 = 0; d0 < activation_dx->shapes[0]; d0++)
            tmp += activation_dx->scalas[d0 * db->size + db_index];
        db->scalas[db_index] = tmp;
    }
}
*/

/// 완전연결층의 순전파 계산을 합니다. 
void cnn_comput_fully_connected_layer_forward(struct Tensor* x, struct Tensor* w, struct Tensor* b, struct Tensor* out, int index, int max_index){
    int out_index = index * out->size / max_index;
    int out_index_max = (index + 1) * out->size / max_index;
    int product_index = 0;
    int pass_x_index = 0;
    int pass_w_index = 0;
    float tmp = 0.0f;
    for (; out_index < out_index_max; out_index++){
        pass_x_index = out_index / b->shapes[0] * w->shapes[0];
        pass_w_index = out_index % b->shapes[0];

        tmp = 0;
        for(product_index = 0; product_index < w->shapes[0]; product_index++)
            tmp += x->scalas[pass_x_index + product_index] * w->scalas[pass_w_index + b->shapes[0] * product_index];
        out->scalas[out_index] = tmp + b->scalas[pass_w_index];
    }
}
void cnn_comput_fully_connected_layer_backward(struct Tensor* dout, struct Tensor* w, struct Tensor* dx, int index, int max_index){
    int dx_index = index * dx->size / max_index;
    int dx_index_max = (index + 1) * dx->size / max_index;
    int product_index = 0;
    int pass_dout_index = 0;
    int pass_w_index = 0;
    int w_row = w->size / w->shapes[0];

    float tmp = 0.0f;

    for (; dx_index < dx_index_max; dx_index++){
        pass_dout_index = dx_index / w->shapes[0] * w_row;
        pass_w_index = dx_index % w->shapes[0] * w_row;

        tmp = 0;
        for (product_index = 0; product_index < w_row; product_index++){
            tmp += dout->scalas[pass_dout_index + product_index] * w->scalas[pass_w_index + product_index];
        }
        dx->scalas[dx_index] = tmp;
    }
}
void cnn_comput_fully_connected_layer_dw(struct Tensor* dout, struct Tensor* x, struct Tensor* dw, int index, int max_index){
    int dw_index = index * dw->size / max_index;
    int dw_index_max = (index  + 1) * dw->size / max_index;
    int dw1 = 0;
    int dw0 = 0;
    int x0 = 0;

    int dw_row = dw->size / dw->shapes[0];
    int x_row = x->size / x->shapes[0];

    float tmp = 0.0f;

    for(; dw_index < dw_index_max; dw_index++){
        dw1 = dw_index % dw_row;
        dw0 = dw_index / dw_row;
        tmp = 0;
        for(x0 = 0; x0 < x->shapes[0]; x0++)
            tmp += x->scalas[x0 * x_row + dw0] * dout->scalas[x0 * dw_row + dw1];
        dw->scalas[dw_index] = tmp;
    }
}
void cnn_comput_fully_connected_layer_db(struct Tensor* dout, struct Tensor* db, int index, int max_index){
    int db_index = index * db->size / max_index;
    int db_index_max = (index + 1) * db->size / max_index;
    int d0 = 0;

    float tmp = 0.0f;

    for(; db_index < db_index_max; db_index++){
        tmp = 0;
        for(d0 = 0; d0 < dout->shapes[0]; d0++)
            tmp += dout->scalas[d0 * db->size + db_index];
        db->scalas[db_index] = tmp;
    }
}