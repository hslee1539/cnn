#pragma once
#include "./struct/tensor_module.h"

void tensor_append_column(struct Tensor* *tensors, int tensors_length, struct Tensor* out);
void tensor_split_column(struct Tensor* tensor, struct Tensor* *out, int out_length);
void tensor_scalas_memcpy(struct Tensor* source, struct Tensor* out);
void tensor_scalas_memcpy_option(struct Tensor* source, struct Tensor* out, int source_index, int out_index, int length);