#include <string.h>
#include "./tool_module.h"


void tensor_append_column(struct Tensor **tensors, int tensors_length, struct Tensor *out){
    int out_index = 0;
    int col_max = 0;
    for(int row = 0, row_max = out->shapes[0]; row < row_max; row++){
        for(int tensor_index = 0; tensor_index < tensors_length; tensor_index++){
            col_max = tensors[tensor_index]->size / row_max;
            tensor_scalas_memcpy_option(tensors[tensor_index], out, col_max * row, out_index, col_max);
            out_index += col_max;
        }
    }
}
void tensor_split_column(struct Tensor *tensor, struct Tensor **out, int out_length){
    int tensor_index = 0;
    int col_max;
    for(int row = 0, row_max = tensor->shapes[0]; row < row_max; row++){
        for(int out_index = 0; out_index < out_length; out_index++){
            col_max = out[out_index]->size / row_max;
            tensor_scalas_memcpy_option(tensor, out[out_index], tensor_index, col_max * row, col_max);
            out_index += col_max;
        }
    }
}

void tensor_scalas_memcpy(struct Tensor *source, struct Tensor *out){
    memcpy(out->scalas, source->scalas, sizeof(float) * out->size);
}

void tensor_scalas_memcpy_option(struct Tensor* source, struct Tensor* out, int source_index, int out_index, int length){
    memcpy(source->scalas + source_index, out->scalas + out_index, sizeof(float) * length);
}